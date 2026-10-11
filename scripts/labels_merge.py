"""Merge every label we have into one multi-class training table: data/labels_v1/labels_all.csv / .gpkg

Classes: Rice, Sugarcane, Cassava, Fallow, Tree, Other.
Sources (high / medium confidence only):
  - GCPs (data/gcps/eda/gcp_verdicts.csv, usable_for_training): Rice, Tree, Tree (young) -> Tree, Sugarcane, Cassava,
    Bare -> Fallow-check (see below)
  - blind audits of rice v1 / v2 / v3: subtype rice -> Rice, tree -> Tree, sugarcane -> Sugarcane,
    upland_crop -> Cassava (corn is practically absent here), built / water -> Other,
    fallow_grass / idle_paddy -> Fallow-check; mixed / other -> dropped
  - multi-class candidate labels (data/labels_v1/labels/batch_*.csv): label as given
Fallow-check = the client's definition: something grew in the previous season. A label only becomes Fallow if the
pixel's cloud-free Sep-Nov 2025 NDVI median is >= 0.45; otherwise it is dropped (never-cropped idle land is not fallow).
Points closer than 30 px (300 m) to another source's point keep both (they are different fields); exact duplicates of
the same pixel keep the first.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import glob, sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data/labels_v1"
SUB = {"rice": "Rice", "tree": "Tree", "sugarcane": "Sugarcane", "upland_crop": "Cassava", "built": "Other",
       "water": "Other", "fallow_grass": "FallowCheck", "idle_paddy": "FallowCheck"}


def sepnov25_ndvi(rows, cols):
    st = PROJ / "data/stacks"
    with rasterio.open(st / "S2_NDVI.tif") as r:
        d = pd.to_datetime(list(r.descriptions))
        idx = [i + 1 for i, x in enumerate(d) if pd.Timestamp("2025-09-01") <= x <= pd.Timestamp("2025-11-30")]
        nd = np.array([r.read(idx, window=Window(c, rw, 1, 1)).ravel() for rw, c in zip(rows, cols)], "float32") / 1e4
    with rasterio.open(st / "S2_CLEAR.tif") as r:
        cl = np.array([r.read(idx, window=Window(c, rw, 1, 1)).ravel() for rw, c in zip(rows, cols)]) == 1
    nd[~cl] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(nd, 1)


def main():
    parts = []
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    g = g[g.usable_for_training]
    gm = {"Rice": "Rice", "Tree": "Tree", "Tree (young)": "Tree", "Sugarcane": "Sugarcane", "Cassava": "Cassava", "Bare": "FallowCheck"}
    parts.append(pd.DataFrame(dict(sample_id=g.gcp_id, row=g.row, col=g.col, cls=g.class_2026.map(gm), source="gcp",
                                   confidence="verified", evidence=g.final_verdict)))
    for f in sorted(glob.glob(str(PROJ / "data/model/rice_v*/audit/audit_results.csv"))):
        a = pd.read_csv(f); v = Path(f).parts[-3]
        a = a[a.label.isin(["Rice", "Non-rice"]) & a.confidence.isin(["high", "medium"])]
        parts.append(pd.DataFrame(dict(sample_id=v + "_" + a.audit_id, row=a.row, col=a.col, cls=a.subtype.map(SUB),
                                       source="audit_" + v, confidence=a.confidence, evidence=a.evidence)))
    key = pd.read_csv(OUT / "key/candidates_key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((OUT / "labels").glob("batch_*.csv"))], ignore_index=True)
    lab = lab.apply(lambda c: c.str.strip() if c.dtype == object else c)
    c = key.merge(lab, on="audit_id")
    c = c[c.confidence.isin(["high", "medium"]) & ~c.label.isin(["Unsure"])]
    parts.append(pd.DataFrame(dict(sample_id="cand_" + c.audit_id, row=c.row, col=c.col, cls=c.label, source="candidates",
                                   confidence=c.confidence, evidence=c.evidence, rule_class=c.rule_class)))
    d = pd.concat(parts, ignore_index=True).dropna(subset=["cls"])
    d = d.drop_duplicates(["row", "col"]).reset_index(drop=True)
    chk = d.cls == "FallowCheck"
    if chk.any():
        prev = sepnov25_ndvi(d.row[chk].values, d.col[chk].values)
        d.loc[chk, "cls"] = np.where(prev >= 0.45, "Fallow", None)
        d = d.dropna(subset=["cls"]).reset_index(drop=True)
    grid = __import__("json").loads((PROJ / "data/grid/grid.json").read_text())
    import geopandas as gpd
    x = grid["x0"] + (d.col + 0.5) * 10; y = grid["y0"] - (d.row + 0.5) * 10
    gdf = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(x, y), crs=grid["crs"])
    gdf.to_file(OUT / "labels_all.gpkg", driver="GPKG"); d.to_csv(OUT / "labels_all.csv", index=False)
    print(pd.crosstab(d.cls, d.source, margins=True).to_string())
    if "rule_class" in d:
        cc = d[d.source == "candidates"]
        print("\ncandidate rule vs analyst label:\n", pd.crosstab(cc.rule_class, cc.cls).to_string())


if __name__ == "__main__":
    main()

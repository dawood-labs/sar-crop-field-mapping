"""Blind field review of the Cane-Cassava map: 100 mapped Cane-Cassava fields + 100 mapped not fields.

usage: audit_cane.py sample  cane_v1   -> data/model/cane_v1/review/{key/review_key.csv, points.csv, sheets/*.png, RUBRIC.md}
       audit_cane.py compare cane_v1   -> review/review_results.csv (+ .parquet), review/review_summary.json
Fields = the output polygons of field_map_v5.py (data/model/<ver>/fields/fields_<ver>.parquet), side = polygon class.
One pixel per field, its 3x3 window inside the polygon; probability bins (pixel cane_prob):
  cane side: >= 80 (50), 50-79 (50)      not side: 20-49 (40), 5-19 (30), 0-4 (30)
Polygons holding any earlier label are skipped; pixels >= 100 m from earlier labels and >= 200 m from each other.
IDs (R<k>_001..) are shuffled. Sheets are the same blind sheets as the rice audits (audit_sheets.sheet); analysts never
see the key or any model output.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
MODE, VER = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], "rice_v5"]
import audit_sheets as asx  # noqa: E402
from safe_resources import workers  # noqa: E402

M = PROJ / "data/model" / VER
A = M / "review"
K = int(VER.split("_v")[1])
BINS = [("cane_high", 80, 100, 50), ("cane_mid", 50, 79, 50), ("not_near", 20, 49, 40), ("not_mid", 5, 19, 30), ("not_low", 0, 4, 30)]
if K >= 2:     # map rule from cane_v2 on: prob >= 40 and rice v5 < 70; "not_guard" = prob >= 40 but stopped by the rice guard
    BINS = [("cane_high", 70, 100, 50), ("cane_mid", 40, 69, 50), ("not_guard", 40, 100, 20), ("not_near", 20, 39, 30),
            ("not_mid", 5, 19, 25), ("not_low", 0, 4, 25)]
ACRE = 100 / 4046.856
RUBRIC = """# Analyst rubric – is this 10 m pixel CANE-CASSAVA (sugarcane or cassava / upland field crop) in 2025/26?

Area: the study area (a rainfed lowland rice district). Period on the sheets: Sep 2025 – Oct 2026. You see only the
evidence, never any model output. The pixel sits INSIDE a field (red / yellow box); judge what that field is.
Sugarcane and cassava (and similar upland field crops) are ONE class here: `Cane-Cassava`.

## How to read a sheet
- Panel 1: black dots = this pixel's cloud-free Sentinel-2 NDVI. Dashed = medians of VERIFIED ground points: green rice,
  brown tree crops, purple sugarcane. Orange band = dry season (Jan–Apr 2026), blue band = monsoon (Jun–Aug 2026).
  Almost no cloud-free images May – mid-Sep 2026.
- Panels 2–3: Sentinel-1 radar VH per track (3x3 pixels), thin line VV; dotted line -20 dB; gray bars = rain.
- 10 small chips B5/B3/B2 on cloud-free dates; 2 context chips (1.28 km), in B8/B4/B3 vegetation is red.

## What each class looks like here
- **Sugarcane (Cane-Cassava):** 12-month crop. Green and tall Sep 2025 – Jan 2026 (NDVI 0.6–0.85), cut Dec – Apr
  (sharp NDVI drop, often Jan–Mar, later than the rice harvest), then regrows DURING the dry season (ratoon: NDVI rising
  Feb–May without flooding; VH rising from Jan/Feb), tall by Aug–Oct 2026 (VH -15…-12 dB). Never standing water.
  Large rectangular blocks. A new cane field: bare Nov–Feb, planted Jan–Apr, steady rise through the dry season.
- **Cassava / upland field crops (Cane-Cassava):** harvested Nov–Mar (often later than rice), replanted Mar–Jun,
  slower, patchier canopy; no standing water; usually slightly higher, sloping land, larger fields than paddies.
- **Rice (Not):** harvested Oct–Dec 2025, bare all dry season with no regrowth, flat bunded paddies; 2026: water in
  Jun–Aug (VH <= -20 dB) or no water but a canopy rise Jul–Oct (direct-seeded).
- **Tree crops / forest (Not):** green all year (rubber dips Jan–Feb), VH stable -16…-13 dB, dark textured canopy.
- **Fallow / grass / idle paddy, built, water (Not).**
- **Trap:** bare dry soil in Feb–Apr gives low VH too; VH bumps after rain are not regrowth – check NDVI.

## What to write (one row per sheet)
review_id,label,subtype,confidence,evidence
- label: `Cane-Cassava` | `Not-cane-cassava` | `Unsure`
- subtype: sugarcane | cassava_upland | rice | tree | fallow_grass | idle_paddy | built | water | mixed | other
- confidence: high | medium | low
- evidence: one sentence with the decisive facts (dates, NDVI, dB, chips)
Judge each pixel on its own. Do not guess what any model would say.
"""


def earlier_labels():
    out = [pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")[["row", "col"]],
           pd.read_csv(PROJ / "data/labels_v1/key/candidates_key.csv")[["row", "col"]],
           pd.read_parquet(M / "samples.parquet")[["row", "col"]]]
    out += [pd.read_csv(k)[["row", "col"]] for k in (PROJ / "data/model").glob("rice_v*/audit/key/audit_key.csv")]
    out += [pd.read_csv(k)[["row", "col"]] for k in (PROJ / "data/model").glob("cane_v*/review/key/review_key.csv")]
    return pd.concat(out).drop_duplicates().to_numpy()


def sample():
    import geopandas as gpd
    from rasterio import features
    from scipy import ndimage
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(1000 + K)
    prob = rasterio.open(M / "cane_prob.tif").read(1)
    with rasterio.open(PROJ / "data/grid/index.tif") as g:
        inside = g.read(2).astype(bool); tr = g.transform
    f = gpd.read_parquet(M / "fields" / f"fields_{VER}.parquet", columns=["poly_id", "class", "geometry"])
    pid = features.rasterize(zip(f.geometry, f.poly_id), out_shape=prob.shape, transform=tr, fill=0, dtype="int32")
    cls = np.zeros(pid.max() + 1, dtype=np.int8); cls[f.poly_id.values] = (f["class"] == "Cane-Cassava").astype(np.int8).values
    interior = (pid > 0) & (ndimage.maximum_filter(pid, 3) == pid) & (ndimage.minimum_filter(pid, 3) == pid) & inside & (prob != 255)
    old = earlier_labels()
    bad = set(np.unique(pid[old[:, 0], old[:, 1]])) - {0}
    rr, cc = np.nonzero(interior); p = pid[rr, cc]
    keep = ~np.isin(p, list(bad)) & (cKDTree(old).query(np.c_[rr, cc])[0] >= 10)
    rr, cc, p = rr[keep], cc[keep], p[keep]
    order = rng.permutation(len(rr)); _, first = np.unique(p[order], return_index=True); pick = order[first]
    c = pd.DataFrame({"row": rr[pick], "col": cc[pick], "poly_id": p[pick]})
    c["side"] = np.where(cls[c.poly_id] == 1, "cane", "not"); c["model_prob"] = prob[c.row, c.col]
    print(f"eligible polygons {len(c):,} (cane {int((c.side == 'cane').sum()):,}); skipped {len(bad):,} with earlier labels", flush=True)
    rows, pts = [], []
    for name, lo, hi, n in BINS:
        side = name.split("_")[0]
        q = c[(c.side == side) & (c.model_prob >= lo) & (c.model_prob <= hi)].sample(frac=1, random_state=int(rng.integers(1e9)))
        got = 0
        for r in q.itertuples():
            if pts and cKDTree(pts).query((r.row, r.col))[0] < 20:
                continue
            pts.append((r.row, r.col)); rows.append(dict(row=r.row, col=r.col, poly_id=r.poly_id, side=side, model_prob=r.model_prob, bin=name)); got += 1
            if got >= n:
                break
        print(f"  {name}: {len(q):,} polygons -> {got}", flush=True)
    d = pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)
    d.insert(0, "review_id", [f"R{K}_{i + 1:03d}" for i in range(len(d))])
    d["model_class"] = np.where(d.side == "cane", "Cane-Cassava", "Not-cane-cassava")
    from pyproj import Transformer
    _, _, t, crs = asx.ex._grid()
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform((t.c + (d.col + 0.5) * 10).values, (t.f - (d.row + 0.5) * 10).values)
    d["lon"], d["lat"] = lon, lat
    return d


def run_sample():
    for s in ("sheets", "key", "labels"):
        (A / s).mkdir(parents=True, exist_ok=True)
    (A / "RUBRIC.md").write_text(RUBRIC)
    d = sample()
    d.to_csv(A / "key/review_key.csv", index=False)
    d[["review_id", "lon", "lat"]].to_csv(A / "points.csv", index=False)
    print(d.groupby("bin").size().to_dict(), flush=True)
    asx.A = A
    T = asx.templates()
    from concurrent.futures import ProcessPoolExecutor
    n = workers(1.5 * 2**30)
    s = d.rename(columns={"review_id": "audit_id"})
    with ProcessPoolExecutor(n) as pool:
        list(pool.map(asx._draw, [s.iloc[i::n] for i in range(n)], [T] * n))
    print("sheets:", len(list((A / "sheets").glob("R*.png"))))


def run_compare():
    key = pd.read_csv(A / "key/review_key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((A / "labels").glob("batch_*.csv"))], ignore_index=True)
    for col in ("review_id", "label", "subtype", "confidence"):
        lab[col] = lab[col].astype(str).str.strip()
    d = key.merge(lab, on="review_id", how="left")
    d.to_csv(A / "review_results.csv", index=False)
    import geopandas as gpd
    gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_parquet(A / "review_results.parquet")
    dec = d[d.label.isin(["Cane-Cassava", "Not-cane-cassava"])]
    hm = dec[dec.confidence.isin(["high", "medium"])]
    per_bin = d.groupby("bin").apply(lambda q: pd.Series({"n": len(q), "cane": int((q.label == "Cane-Cassava").sum()),
                                                          "not": int((q.label == "Not-cane-cassava").sum()), "unsure": int((q.label == "Unsure").sum())}),
                                     include_groups=False)
    # area-weighted over the map: each bin weighted by its pixels in the map (polygon side x pixel prob bin)
    prob = rasterio.open(M / "cane_prob.tif").read(1)
    sv = rasterio.open(M / "fields" / "cane_map_sieve.tif").read(1)
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool) & (prob != 255)
    wts = {}
    for name, lo, hi, _ in BINS:
        side = (sv == 1) if name.startswith("cane") else (sv == 0)
        wts[name] = float(((prob >= lo) & (prob <= hi) & side & inside).sum() * ACRE)
    rows = []
    for name, q in hm.groupby("bin"):
        share = float((q.label == "Cane-Cassava").mean())
        rows.append(dict(bin=name, map_acres=wts[name], n=len(q), cane_share=share, est_cane_acres=wts[name] * share))
    t = pd.DataFrame(rows)
    mapped = t[t.bin.str.startswith("cane")]
    est = {"map_cane_acres_in_bins": float(mapped.map_acres.sum()), "est_true_cane_acres_in_bins": float(t.est_cane_acres.sum()),
           "area_weighted_precision": float(mapped.est_cane_acres.sum() / max(mapped.map_acres.sum(), 1)),
           "area_weighted_recall": float(mapped.est_cane_acres.sum() / max(t.est_cane_acres.sum(), 1))}
    agree = float((hm.label == hm.model_class).mean()) if len(hm) else float("nan")
    s = {"n": int(len(d)), "missing": int(d.label.isna().sum()), "labels": d.label.value_counts().to_dict(),
         "confidence": d.confidence.value_counts().to_dict(), "agreement_high_medium": agree, "n_high_medium": int(len(hm)),
         "per_bin": per_bin.reset_index().to_dict("records"), "area_weighted": est, "per_bin_area": t.round(3).to_dict("records"),
         "errors_by_subtype": hm[hm.label != hm.model_class].groupby(["model_class", "subtype"]).size().astype(int).to_dict()}
    s["errors_by_subtype"] = {f"{a} -> {b}": v for (a, b), v in s["errors_by_subtype"].items()}
    (A / "review_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(per_bin.to_string()); print(json.dumps({k: s[k] for k in ("labels", "agreement_high_medium", "n_high_medium", "area_weighted", "errors_by_subtype")},
                                                 indent=1, default=lambda v: round(float(v), 3)))


if __name__ == "__main__":
    run_sample() if MODE == "sample" else run_compare()

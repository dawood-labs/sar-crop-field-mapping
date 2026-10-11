"""S2 QC / correction of the rice + cane-cassava deliverable (user approved S2 for correction, 2026-10-09).

Why: the user found split pieces of one uniform cane field labelled half rice (e.g. 27324 / 48828, 17337 / 50072).
Test on verified labels (pixel 3x3, 5-fold CV): NDVI clear medians of Apr-May + Aug-Oct alone separate cane+cassava
from pathlib import Path
from rice poorly (bal. acc 0.66); with Oct-Nov 2025, Dec-Jan and Feb-Mar added: 0.86 (sugarcane vs rice 0.90);
at p >= 0.8 the S2 call is cane with P 0.92 (11 of 403 rice labels wrong). Dec-Jan NDVI is the key (cane 0.55, rice 0.33).
Steps: NDVI windows (median + max of cloud-free dates) -> RandomForest on all verified labels -> mean p_cane per field
(zonal mean of the features over each polygon) -> relabel only confident disagreements:
  Rice with p_cane >= 0.8 -> Cane-Cassava;  Cane-Cassava with p_cane <= 0.1 -> Rice
-> split pieces of one parent (same uid) that touch across the cut and now share a class are merged back
-> separate 0.1 m + the deliverable QC -> data/deliverables/ (v1 moved to data/deliverables/v1/).
Outputs: data/qc/s2_ndvi_windows.tif (cache), data/deliverables/<NAME>_s2qc_changes.csv, the v2 deliverable files.
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, shutil, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
import topology as tp  # noqa: E402
import deliver_fields as dlv  # noqa: E402

WIN = {"octnov25": ("2025-10-01", "2025-11-30"), "decjan": ("2025-12-01", "2026-01-31"), "febmar": ("2026-02-01", "2026-03-31"),
       "aprmay": ("2026-04-01", "2026-05-31"), "augoct": ("2026-08-01", "2026-10-09")}
CACHE = PROJ / "data/qc/s2_ndvi_windows.tif"
TO_CANE, TO_RICE = 0.8, 0.1
D = dlv.OUT
ACRE_M2 = 4046.856


def windows():
    if CACHE.exists():
        with rasterio.open(CACHE) as r:
            return {n: r.read(i + 1) for i, n in enumerate(r.descriptions)}
    with rasterio.open(PROJ / "data/stacks/S2_NDVI.tif") as r:
        d = pd.to_datetime(list(r.descriptions))
    def comp(w):
        a, b = WIN[w]; k = [i + 1 for i, x in enumerate(d) if pd.Timestamp(a) <= x <= pd.Timestamp(b)]
        cl = rasterio.open(PROJ / "data/stacks/S2_CLEAR.tif").read(k) == 1
        x = rasterio.open(PROJ / "data/stacks/S2_NDVI.tif").read(k).astype("f4") / 1e4; x[~cl] = np.nan
        with np.errstate(all="ignore"):
            return {f"ndvi_med_{w}": np.nanmedian(x, 0), f"ndvi_max_{w}": np.nanmax(x, 0)}
    out = {}
    with ThreadPoolExecutor(len(WIN)) as p:
        for r_ in p.map(comp, WIN):
            out.update(r_)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    prof = rasterio.open(PROJ / "data/grid/index.tif").profile
    with rasterio.open(CACHE, "w", **dict(prof, count=len(out), dtype="float32", nodata=np.nan, compress="deflate")) as o:
        for i, (n, a) in enumerate(sorted(out.items()), 1):
            o.write(a.astype("f4"), i); o.set_band_description(i, n)
    return dict(sorted(out.items()))


def labels():
    L = []
    for v in (1, 2):
        a = pd.read_csv(PROJ / f"data/model/cane_v{v}/review/review_results.csv")
        a = a[a.confidence.isin(["high", "medium"]) & a.label.isin(["Cane-Cassava", "Not-cane-cassava"])]
        L.append(pd.DataFrame({"row": a.row, "col": a.col, "cls": a.subtype}))
    for v in (1, 2, 3, 4):
        a = pd.read_csv(PROJ / f"data/model/rice_v{v}/audit/audit_results.csv"); a = a[a.confidence.isin(["high", "medium"])]
        L.append(pd.DataFrame({"row": a.row, "col": a.col, "cls": a.subtype}))
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv"); g = g[g.usable_for_training]
    L.append(pd.DataFrame({"row": g.row, "col": g.col, "cls": g.class_2026.str.lower()}))
    L = pd.concat(L).drop_duplicates(["row", "col"])
    L["cls"] = L.cls.replace({"upland_crop": "cassava", "cassava_upland": "cassava"})
    return L[L.cls.isin(["rice", "sugarcane", "cassava"])].reset_index(drop=True)


def main():
    import geopandas as gpd, shapely
    from rasterio import features
    from scipy.ndimage import uniform_filter
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    Wd = windows(); names = list(Wd)
    print(f"S2 windows {len(names)} bands ({time.time() - t0:.0f} s)", flush=True)
    # training: 3x3 means at the labels (as tested)
    L = labels()
    X = np.zeros((len(L), len(names)), "f4")
    for j, n in enumerate(names):
        a = Wd[n]; ok = np.isfinite(a)
        with np.errstate(all="ignore"):
            m = uniform_filter(np.where(ok, a, 0), 3) / uniform_filter(ok.astype("f4"), 3)
        X[:, j] = m[L.row, L.col]
    y = (L.cls != "rice").astype(int).values
    rf = RandomForestClassifier(400, min_samples_leaf=3, class_weight="balanced", n_jobs=os.cpu_count(), random_state=0)
    rf.fit(np.nan_to_num(X, nan=-1), y)
    # fields: zonal mean of every feature over each polygon (polygons without a pixel centre: their centroid pixel)
    g = gpd.read_parquet(D / f"{dlv.NAME}.parquet")
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        tr, shape = r.transform, r.shape
    fid = features.rasterize(zip(g.geometry, g.field_id), out_shape=shape, transform=tr, fill=0, dtype="int32")
    n = len(g) + 1; F = np.full((len(g), len(names)), np.nan, "f4")
    m = fid > 0
    rp = g.geometry.representative_point(); rr, cc = rasterio.transform.rowcol(tr, rp.x.values, rp.y.values)
    for j, nme in enumerate(names):
        a = Wd[nme]; ok = m & np.isfinite(a)
        s = np.bincount(fid[ok], weights=a[ok], minlength=n); c = np.bincount(fid[ok], minlength=n)
        with np.errstate(all="ignore"):
            v = (s / c)[g.field_id.values]
        miss = ~np.isfinite(v); v[miss] = a[np.clip(rr, 0, shape[0] - 1), np.clip(cc, 0, shape[1] - 1)][miss]
        F[:, j] = v
    g["p_cane_s2"] = rf.predict_proba(np.nan_to_num(F, nan=-1))[:, 1].round(3)
    old = g["class"].copy()
    to_cane = (old == "Rice") & (g.p_cane_s2 >= TO_CANE); to_rice = (old == "Cane-Cassava") & (g.p_cane_s2 <= TO_RICE)
    g.loc[to_cane, "class"] = "Cane-Cassava"; g.loc[to_rice, "class"] = "Rice"
    g["s2_qc"] = np.where(to_cane, "rice->cane", np.where(to_rice, "cane->rice", "kept"))
    print(f"relabel: rice->cane {int(to_cane.sum())} ({g.area_acres[to_cane].sum():.0f} ac), cane->rice {int(to_rice.sum())} "
          f"({g.area_acres[to_rice].sum():.0f} ac) ({time.time() - t0:.0f} s)", flush=True)

    # merge split pieces of one parent that now share a class and face each other across the cut
    sp = g[g.source == "split"]
    G = np.asarray(g.geometry.values, dtype=object)
    pairs = tp.near_pairs(G[sp.index.values], 2 * tp.GAP + 0.05)
    a_, b_ = sp.index.values[pairs[0]], sp.index.values[pairs[1]]
    same = (g.uid.values[a_] == g.uid.values[b_]) & (g["class"].values[a_] == g["class"].values[b_]) & (a_ < b_)
    a_, b_ = a_[same], b_[same]
    parent = np.arange(len(g))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in zip(a_, b_):
        parent[find(i)] = find(j)
    root = np.array([find(i) for i in range(len(g))])
    groups = pd.Series(np.arange(len(g))).groupby(root).apply(list)
    groups = [ix for ix in groups if len(ix) > 1]
    kw = dict(quad_segs=2, join_style="mitre", mitre_limit=5.0)
    merged_from = {}
    for ix in groups:                               # close the 0.1 m cut between the pieces, nothing else
        u = shapely.union_all(shapely.buffer(G[ix], tp.GAP, **kw), grid_size=tp.GRID)
        G[ix[0]] = tp.fill_small_holes([shapely.make_valid(shapely.buffer(u, -tp.GAP, **kw))])[0]   # no slit left on the cut
        for k in ix[1:]:
            G[k] = shapely.Polygon()
        merged_from[ix[0]] = ix
    g["geometry"] = G
    g["merged_pieces"] = 1
    for k, ix in merged_from.items():
        g.loc[k, "merged_pieces"] = len(ix)
        g.loc[k, "p_cane_s2"] = float(np.average(g.loc[ix, "p_cane_s2"], weights=g.loc[ix, "area_acres"]))
    changes = g.loc[(g.s2_qc != "kept") | (g.merged_pieces > 1), ["field_id", "uid", "s2_qc", "merged_pieces", "p_cane_s2", "area_acres"]].copy()
    changes["old_class"] = old[changes.index]; changes["new_class"] = g.loc[changes.index, "class"]
    g = g[~g.is_empty].reset_index(drop=True)
    print(f"split pieces merged back: {len(groups)} groups ({sum(len(x) for x in groups)} pieces) ({time.time() - t0:.0f} s)", flush=True)

    # final topology + QC, same as the deliverable
    G = tp.separate(np.asarray(g.geometry.values, dtype=object))
    g["geometry"] = G
    g = g[~g.is_empty].explode(index_parts=False).reset_index(drop=True)
    g = g[(g.geom_type == "Polygon") & (g.area >= 100.0)].reset_index(drop=True)
    aoi = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(g.crs).union_all()
    g["predicted"] = g["class"].map(dlv.PREDICTED).astype("int32")
    g["area_acres"] = (g.area / ACRE_M2).round(4)
    g = g.sort_values(["predicted", "area_acres"], ascending=[False, False]).reset_index(drop=True)
    g["field_id_v1"] = g.field_id
    g["field_id"] = np.arange(1, len(g) + 1, dtype="int32")
    rep, fail = dlv.qc(g, aoi)
    rep["dissolve_ok"] = rep["intersecting_or_touching_pairs"] == 0
    print("QC:", json.dumps({k: v for k, v in rep.items() if k != "crs"}, default=float), "FAIL" if fail else "passed", flush=True)
    if fail:
        raise SystemExit(f"QC failed: {fail}")

    v1 = D / "v1"; v1.mkdir(exist_ok=True)
    for f in D.glob(f"{dlv.NAME}*"):
        if f.is_file():
            shutil.move(str(f), v1 / f.name)
    shutil.rmtree(D / f"{dlv.NAME}_shp", ignore_errors=True)
    changes.to_csv(D / f"{dlv.NAME}_s2qc_changes.csv", index=False)
    g[["field_id", "predicted", "class", "area_acres", "p_cane_s2", "s2_qc", "merged_pieces", "field_id_v1", "class_frac", "source",
       "uid", "Shp_Label", "Confidence", "geometry"]].to_parquet(D / f"{dlv.NAME}.parquet", compression="zstd", write_covering_bbox=True)
    shp_dir = D / f"{dlv.NAME}_shp"; shp_dir.mkdir(exist_ok=True)
    g[["field_id", "predicted", "class", "area_acres", "geometry"]].rename(columns={"area_acres": "area_ac"}).to_file(shp_dir / f"{dlv.NAME}.shp", encoding="UTF-8")
    zp = D / f"{dlv.NAME}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(shp_dir.glob(f"{dlv.NAME}.*")):
            z.write(f, f.name)
    back = gpd.read_file(f"zip://{zp}!{dlv.NAME}.shp")
    rb, fb = dlv.qc(back, aoi)
    rep.update(version="v2 (S2 QC)", relabel={"rice->cane": int(to_cane.sum()), "cane->rice": int(to_rice.sum())},
               merged_groups=len(groups), acres_by_class=g.groupby("class").area_acres.sum().round(1).to_dict(),
               polygons_by_class=g["class"].value_counts().to_dict(),
               shapefile_readback={"features": rb["features"], "invalid": rb["invalid"], "touching": rb["intersecting_or_touching_pairs"], "fail": fb})
    (D / f"{dlv.NAME}_qc.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("relabel", "merged_groups", "acres_by_class", "polygons_by_class", "shapefile_readback")}, default=float))
    if fb:
        raise SystemExit(f"shapefile QC failed: {fb}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

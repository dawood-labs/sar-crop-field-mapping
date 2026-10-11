"""Deliverable v3 (user, 2026-10-09): recover missed cane-cassava with S2 and split sugarcane from cassava.

A. Fields that are in no class yet (clean delineation polygons outside every v2 polygon) get an S2 probability of
   cane-cassava vs EVERYTHING else (rice, tree, fallow, idle paddy, built, water, other); p >= 0.8 -> added as cane.
B. Every cane-cassava field gets p_sugarcane (sugarcane vs cassava/upland) from monthly NDVI: >= 0.5 -> Sugarcane
   (predicted 1), else Cassava (predicted 281) (company class codes v2.0); conf = high if p >= 0.8 or <= 0.2.
   Test on verified labels (5-fold CV): monthly NDVI bal. acc 0.73 (cane recall 0.92, cassava 0.55); confident 43 %
   of labels at 95 %. Rice stays 7.
Features: clear-median NDVI per month Oct 2025 - Oct 2026 (cache data/qc/s2_ndvi_monthly.tif) + the 5 windows of
qc_s2_fields.py (median / max); field value = zonal mean. Then separate 0.1 m + the deliverable QC; v2 -> deliverables/v2/.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, shutil, sys, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
import topology as tp  # noqa: E402
import deliver_fields as dlv  # noqa: E402
import qc_s2_fields as q2  # noqa: E402

MONTHLY = PROJ / "data/qc/s2_ndvi_monthly.tif"
ADD_CANE, MIN_NEW_AC = 0.7, 0.15          # user 2026-10-09: 0.7 (CV P 0.88, R 0.65); was 0.8 (P 0.94, R 0.52)
SPLIT_BIG_AC = 1.0                         # unclassified polygons >= 1 ac not added whole: per-pixel S2 score + straight cuts
PRED = {"Rice": 7, "Sugarcane": 1, "Cassava": 281}
D = dlv.OUT
ACRE_M2 = 4046.856


def monthly():
    if MONTHLY.exists():
        with rasterio.open(MONTHLY) as r:
            return {n: r.read(i + 1) for i, n in enumerate(r.descriptions)}
    with rasterio.open(PROJ / "data/stacks/S2_NDVI.tif") as r:
        d = pd.to_datetime(list(r.descriptions)); x = r.read().astype("f4") / 1e4
    x[rasterio.open(PROJ / "data/stacks/S2_CLEAR.tif").read() != 1] = np.nan
    out = {}
    for m in pd.period_range("2025-10", "2026-10", freq="M"):
        k = [i for i, t in enumerate(d) if t.to_period("M") == m]
        if k:
            with np.errstate(all="ignore"):
                out[f"ndvi_{m}"] = np.nanmedian(x[k], 0)
    prof = rasterio.open(PROJ / "data/grid/index.tif").profile
    with rasterio.open(MONTHLY, "w", **dict(prof, count=len(out), dtype="float32", nodata=np.nan, compress="deflate")) as o:
        for i, (n, a) in enumerate(out.items(), 1):
            o.write(a.astype("f4"), i); o.set_band_description(i, n)
    return out


def labels_all():
    L = []
    for v in (1, 2):
        a = pd.read_csv(PROJ / f"data/model/cane_v{v}/review/review_results.csv")
        a = a[a.confidence.isin(["high", "medium"]) & a.label.isin(["Cane-Cassava", "Not-cane-cassava"])]
        L.append(pd.DataFrame({"row": a.row, "col": a.col, "cls": a.subtype}))
    for v in (1, 2, 3, 4):
        a = pd.read_csv(PROJ / f"data/model/rice_v{v}/audit/audit_results.csv"); a = a[a.confidence.isin(["high", "medium"])]
        L.append(pd.DataFrame({"row": a.row, "col": a.col, "cls": a.subtype}))
    k = pd.read_csv(PROJ / "data/labels_v1/key/candidates_key.csv")
    c = pd.concat([pd.read_csv(f) for f in sorted((PROJ / "data/labels_v1/labels").glob("batch_*.csv"))]).merge(k, on="audit_id")
    c = c[c.confidence.isin(["high", "medium"]) & (c.label != "Unsure")]
    L.append(pd.DataFrame({"row": c.row, "col": c.col, "cls": c.label.str.lower()}))
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv"); g = g[g.usable_for_training]
    L.append(pd.DataFrame({"row": g.row, "col": g.col, "cls": g.class_2026.str.lower()}))
    L = pd.concat(L).drop_duplicates(["row", "col"])
    L["cls"] = L.cls.replace({"upland_crop": "cassava", "cassava_upland": "cassava", "tree (young)": "tree", "fallow_grass": "fallow",
                              "idle_paddy": "fallow", "bare": "other"})
    return L[L.cls.isin(["rice", "sugarcane", "cassava", "tree", "fallow", "built", "water", "other"])].reset_index(drop=True)


def main():
    import geopandas as gpd, shapely
    from rasterio import features
    from scipy.ndimage import uniform_filter
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    t0 = time.time()
    R = {**monthly(), **q2.windows()}; names = list(R)
    print(f"S2 features {len(names)} ({time.time() - t0:.0f} s)", flush=True)
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        tr, shape = r.transform, r.shape
    L = labels_all()
    X = np.zeros((len(L), len(names)), "f4")
    for j, n in enumerate(names):
        a = R[n]; ok = np.isfinite(a)
        with np.errstate(all="ignore"):
            X[:, j] = (uniform_filter(np.where(ok, a, 0), 3) / uniform_filter(ok.astype("f4"), 3))[L.row, L.col]
    X = np.nan_to_num(X, nan=-1)
    rf = lambda: RandomForestClassifier(500, min_samples_leaf=2, class_weight="balanced", n_jobs=os.cpu_count(), random_state=0)  # noqa: E731
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    # A: cane-cassava vs everything else
    yA = L.cls.isin(["sugarcane", "cassava"]).astype(int).values
    pA = cross_val_predict(rf(), X, yA, cv=cv, method="predict_proba")[:, 1]
    hit = pA >= ADD_CANE
    testA = {"n": int(len(yA)), f"precision_at_{ADD_CANE}": float(yA[hit].mean()), f"recall_at_{ADD_CANE}": float(hit[yA == 1].mean()),
             f"false_cane_at_{ADD_CANE}_by_class": L[hit & (yA == 0)].cls.value_counts().to_dict(),
             "non_cane_labels_by_class": L[yA == 0].cls.value_counts().to_dict()}
    print("A (cane-cassava vs all) CV:", json.dumps(testA), flush=True)
    mA = rf().fit(X, yA)
    # B: sugarcane vs cassava
    sb = L.cls.isin(["sugarcane", "cassava"]).values
    yB = (L.cls[sb] == "sugarcane").astype(int).values
    pB = cross_val_predict(rf(), X[sb], yB, cv=cv, method="predict_proba")[:, 1]
    prB = pB >= 0.5; conf = (pB >= 0.8) | (pB <= 0.2)
    testB = {"n": int(sb.sum()), "bal_acc": float((prB[yB == 1].mean() + (~prB[yB == 0]).mean()) / 2),
             "sugarcane_precision": float(yB[prB].mean()), "cassava_precision": float((1 - yB[~prB]).mean()),
             "confident_share": float(conf.mean()), "confident_acc": float((prB[conf] == yB[conf]).mean())}
    print("B (sugarcane vs cassava) CV:", json.dumps(testB), flush=True)
    mB = rf().fit(X[sb], yB)

    def zonal(gdf):
        fid = features.rasterize(zip(gdf.geometry, np.arange(1, len(gdf) + 1)), out_shape=shape, transform=tr, fill=0, dtype="int32")
        n = len(gdf) + 1; m = fid > 0; Fz = np.full((len(gdf), len(names)), np.nan, "f4")
        rp = gdf.geometry.representative_point(); rr, cc = rasterio.transform.rowcol(tr, rp.x.values, rp.y.values)
        rr, cc = np.clip(rr, 0, shape[0] - 1), np.clip(cc, 0, shape[1] - 1)
        for j, nme in enumerate(names):
            a = R[nme]; ok = m & np.isfinite(a)
            s = np.bincount(fid[ok], weights=a[ok], minlength=n); c = np.bincount(fid[ok], minlength=n)
            with np.errstate(all="ignore"):
                v = (s / c)[1:]
            miss = ~np.isfinite(v); v[miss] = a[rr, cc][miss]; Fz[:, j] = v
        return np.nan_to_num(Fz, nan=-1)

    # A: candidates = clean delineation polygons (AOI) minus every v2 polygon (+ 0.1 m), parts >= 0.15 ac
    v2 = gpd.read_parquet(D / "v2" / f"{dlv.NAME}.parquet")
    aoi = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(v2.crs).union_all()
    f = gpd.read_parquet(PROJ / "data/fields/delineation/fields_clean.parquet", columns=["uid", "Shp_Label", "Confidence", "geometry"])
    f = f[f.intersects(aoi)].reset_index(drop=True)
    f["geometry"] = tp.safe_difference(np.asarray(f.geometry.values, dtype=object), np.array([aoi.envelope.buffer(500).difference(aoi)] * len(f), dtype=object))
    G = np.asarray(f.geometry.values, dtype=object); V = np.asarray(v2.geometry.values, dtype=object)
    Vb = tp._threaded(lambda g: shapely.make_valid(shapely.buffer(g, tp.GAP, quad_segs=2)), V)
    b = shapely.bounds(G); pr = shapely.STRtree(Vb).query(shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
    pr = pr[:, tp._threaded(lambda c: shapely.intersects(G[c[:, 0]], Vb[c[:, 1]]), pr.T.copy()).astype(bool)]
    Gc = tp.subtract_neighbours(G, Vb, np.vstack([pr[0], pr[1]])) if pr.shape[1] else G
    f["geometry"] = Gc
    f = f[~f.is_empty].explode(index_parts=False).reset_index(drop=True)
    f = f[(f.geom_type == "Polygon") & (f.area / ACRE_M2 >= MIN_NEW_AC)].reset_index(drop=True)
    f["geometry"] = tp.fill_small_holes(np.asarray(f.geometry.values, dtype=object))
    f["p_cc_s2"] = mA.predict_proba(zonal(f))[:, 1].round(3)
    f.to_parquet(PROJ / "data/qc/unclassified_candidates_v3.parquet", compression="zstd", write_covering_bbox=True)
    new = f[f.p_cc_s2 >= ADD_CANE].copy()
    # big mixed candidates: per-pixel S2 score (3x3 features), straight cuts as in field_map_v5, keep pieces >= ADD_CANE
    import field_map_v5 as fm
    big = f[(f.p_cc_s2 < ADD_CANE) & (f.area / ACRE_M2 >= SPLIT_BIG_AC)].reset_index(drop=True)
    fidb = features.rasterize(zip(big.geometry, np.arange(1, len(big) + 1)), out_shape=shape, transform=tr, fill=0, dtype="int32")
    rr_, cc_ = np.nonzero(fidb)
    Xp = np.zeros((len(rr_), len(names)), "f4")
    for j, nme in enumerate(names):
        a = R[nme]; ok = np.isfinite(a)
        with np.errstate(all="ignore"):
            Xp[:, j] = (uniform_filter(np.where(ok, a, 0), 3) / uniform_filter(ok.astype("f4"), 3))[rr_, cc_]
    pp = mA.predict_proba(np.nan_to_num(Xp, nan=-1))[:, 1]
    order = np.argsort(fidb[rr_, cc_], kind="stable"); ids = fidb[rr_, cc_][order]
    starts = np.searchsorted(ids, np.arange(1, len(big) + 1)); ends = np.searchsorted(ids, np.arange(1, len(big) + 1), side="right")
    pieces = []
    old_min = fm.MINORITY; fm.MINORITY = 0.15                 # a cane block of >= 15 % of a big polygon is worth cutting out
    for k in range(len(big)):
        o = order[starts[k]:ends[k]]
        if len(o) < 4 or (pp[o] >= ADD_CANE).sum() < 9:     # needs >= 9 confident px (0.22 ac) of cane
            continue
        xy = np.c_[tr.c + (cc_[o] + 0.5) * tr.a, tr.f + (rr_[o] + 0.5) * tr.e]
        P = big.geometry.iat[k]
        for gq, n_, nr_, kind in fm.split(P, xy, (pp[o] >= ADD_CANE).astype(int), P):
            if kind == "deleted" or kind == "whole" or n_ == 0:
                continue
            inside = shapely.contains_xy(gq, xy[:, 0], xy[:, 1])
            if inside.sum() and pp[o][inside].mean() >= ADD_CANE and gq.area / ACRE_M2 >= MIN_NEW_AC:
                pieces.append(dict(uid=big.uid.iat[k], Shp_Label=big.Shp_Label.iat[k], Confidence=big.Confidence.iat[k],
                                   p_cc_s2=round(float(pp[o][inside].mean()), 3), geometry=gq))
    fm.MINORITY = old_min
    if pieces:
        new = pd.concat([new, gpd.GeoDataFrame(pieces, crs=f.crs)], ignore_index=True)
    print(f"A: big mixed candidates {len(big):,}; cane pieces cut out of them {len(pieces):,} "
          f"({sum(p['geometry'].area for p in pieces) / ACRE_M2:,.0f} ac)", flush=True)
    new["geometry"] = tp.refine(np.asarray(new.geometry.values, dtype=object))      # same shape clean-up as the rest
    new = new[~new.is_empty]
    print(f"A: {len(f):,} unclassified field pieces ({f.area.sum() / ACRE_M2:,.0f} ac); recovered as cane-cassava: {len(new):,} "
          f"({new.area.sum() / ACRE_M2:,.0f} ac) ({time.time() - t0:.0f} s)", flush=True)

    # combine, B on every cane-cassava polygon
    v2 = v2.rename(columns={"p_cane_s2": "p_cane_vs_rice_s2"})
    new = new.assign(**{"class": "Cane-Cassava", "source": "s2_recovered", "s2_qc": "added", "merged_pieces": 1, "class_frac": np.nan,
                        "field_id_v1": -1, "p_cane_vs_rice_s2": np.nan})
    g = gpd.GeoDataFrame(pd.concat([v2, new.drop(columns="p_cc_s2").assign(p_cc_s2=new.p_cc_s2)], ignore_index=True), crs=v2.crs)
    g = g.rename(columns={"field_id": "field_id_v2"})
    cc = (g["class"] == "Cane-Cassava").values
    g["p_sugarcane"] = np.nan
    g.loc[cc, "p_sugarcane"] = mB.predict_proba(zonal(g[cc]))[:, 1].round(3)
    g.loc[cc, "class"] = np.where(g.loc[cc, "p_sugarcane"] >= 0.5, "Sugarcane", "Cassava")
    g["conf"] = np.where(cc, np.where((g.p_sugarcane >= 0.8) | (g.p_sugarcane <= 0.2), "high", "low"), "")
    print(f"B: {int(cc.sum()):,} cane-cassava fields -> {g.loc[cc, 'class'].value_counts().to_dict()}, high conf {int((g.conf == 'high').sum()):,}", flush=True)

    # topology + QC
    g["geometry"] = tp.separate(np.asarray(g.geometry.values, dtype=object))
    g = g[~g.is_empty].explode(index_parts=False).reset_index(drop=True)
    g = g[(g.geom_type == "Polygon") & (g.area >= 100.0)].reset_index(drop=True)
    g["geometry"] = tp.fill_small_holes(np.asarray(g.geometry.values, dtype=object))
    g["predicted"] = g["class"].map(PRED).astype("int32")
    g["area_acres"] = (g.area / ACRE_M2).round(4)
    g = g.sort_values(["predicted", "area_acres"], ascending=[True, False]).reset_index(drop=True)
    g.insert(0, "field_id", np.arange(1, len(g) + 1, dtype="int32"))
    rep, fail = dlv.qc(g, aoi)
    rep["dissolve_ok"] = rep["intersecting_or_touching_pairs"] == 0
    print("QC:", json.dumps({k: v for k, v in rep.items() if k != "crs"}, default=float), "FAIL" if fail else "passed", flush=True)
    if fail:
        raise SystemExit(f"QC failed: {fail}")

    v2dir = D / "v2"; v2dir.mkdir(exist_ok=True)
    for p in (D.glob(f"{dlv.NAME}*") if not (v2dir / f"{dlv.NAME}.zip").exists() else []):
        if p.is_file():
            shutil.move(str(p), v2dir / p.name)
    shutil.rmtree(D / f"{dlv.NAME}_shp", ignore_errors=True)
    keep = ["field_id", "predicted", "class", "area_acres", "conf", "p_sugarcane", "p_cc_s2", "p_cane_vs_rice_s2", "s2_qc", "source",
            "merged_pieces", "field_id_v2", "field_id_v1", "class_frac", "uid", "Shp_Label", "Confidence", "geometry"]
    g[keep].to_parquet(D / f"{dlv.NAME}.parquet", compression="zstd", write_covering_bbox=True)
    shp_dir = D / f"{dlv.NAME}_shp"; shp_dir.mkdir(exist_ok=True)
    g[["field_id", "predicted", "class", "area_acres", "conf", "geometry"]].rename(columns={"area_acres": "area_ac"}) \
        .to_file(shp_dir / f"{dlv.NAME}.shp", encoding="UTF-8")
    zp = D / f"{dlv.NAME}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(shp_dir.glob(f"{dlv.NAME}.*")):
            z.write(p, p.name)
    back = gpd.read_file(f"zip://{zp}!{dlv.NAME}.shp")
    rb, fb = dlv.qc(back, aoi)
    rep.update(version="v3", test_cane_vs_all=testA, test_sugarcane_vs_cassava=testB, recovered_fields=int(len(new)),
               recovered_acres=float(new.area.sum() / ACRE_M2), acres_by_class=g.groupby("class").area_acres.sum().round(1).to_dict(),
               polygons_by_class=g["class"].value_counts().to_dict(), conf_by_class=g[g.conf != ""].groupby(["class", "conf"]).size().to_dict(),
               predicted_codes=PRED, shapefile_readback={"features": rb["features"], "invalid": rb["invalid"],
                                                          "touching": rb["intersecting_or_touching_pairs"], "fail": fb})
    rep["conf_by_class"] = {f"{a}/{b}": v for (a, b), v in rep["conf_by_class"].items()}
    (D / f"{dlv.NAME}_qc.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("recovered_fields", "recovered_acres", "acres_by_class", "polygons_by_class", "conf_by_class",
                                           "shapefile_readback")}, default=float))
    if fb:
        raise SystemExit(f"shapefile QC failed: {fb}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

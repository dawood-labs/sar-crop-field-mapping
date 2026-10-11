"""Rice vs non-rice, model v6 = v5 + 6 extra SAR features against sugarcane mapped as rice (data/features/
s1_features_v6_extra.tif: VH Jan->Mar rise = ratoon regrowth, Dec-Jan vs Feb-Apr drop = late cut, dry-season
amplitude; per track) + 2 user-checked sugarcane pixels (1348570, 988937; user check of v5, 2026-10-09). SAR-only.
Ablation: the same samples and the same CV folds are run twice, with v5's 86 features and with v6's 92, so the
out-of-fold numbers show only the effect of the new features (field audit area-weighted like v5's report).
Outputs: data/model/rice_v6/ (same files as v5) + compare_v5_v6.json, change_v5_v6.tif.
--- v5 notes (unchanged) ---
Rice vs non-rice, model v5 = v4 + the high/medium labels of the FIELD audit of v4 (400 pixels, one per FTW field,
never on a field edge: 200 mapped rice + 200 mapped non-rice; data/model/rice_v4/audit). SAR-only as v4.
Honest comparison on the 400 field-audit pixels: v3 and v4 never saw them (published maps), v5 = out-of-fold.
Outputs: data/model/rice_v5/ (same files as v4) + compare_v4_v5.json, change_v4_v5.tif, test_field_audit.csv.
--- v4 notes (unchanged) ---
Rice vs non-rice, model v4 (Sentinel-1 + terrain), compared with v3.

Why (QC of v3 on 255 independent labels): errors are almost all field-EDGE pixels, and the missed rice is mostly
direct-seeded / late rice (no standing-water dip, weak VH rise). Changes from v3:
  1. features data/features/s1_features_v4.tif (84 bands, 3x3 window instead of 5x5, 1x1 VH Jun-Sep, per-date
     nwet / vhmin / vhrise / vhlate / vhdry_trend) + elevation / slope;
  2. training = GCP (3x3) + audits v1/v2/v3 high/medium Rice/Non-rice + data/labels_v1 candidate labels
     (high/medium; Rice -> 1, Sugarcane/Cassava/Fallow/Tree/Other -> 0) + WorldCover rules as v3
     + Unsure pixels (audits v1-v3 and candidates) that pass the DIRECT-SEEDED RICE rule -> rice (others dropped);
  3. CV groups = 500 m blocks for every non-GCP sample (less spatial leakage than 1 group per pixel).
SAR ONLY (user, 2026-10-09): no S2 / NDVI anywhere in the model, neither as a feature nor to pick samples; S2 is only
for verification and analysis. All thresholds below are means of both tracks, from the v4 features (3x3, dB), set from
the SAR values of the labelled pixels (rice vs tree / built / sugarcane / fallow).
DSR rule: harvest Oct-Dec 2025 (VH Nov 2025 - VH Jan 2026 >= 2 dB), bare dry season (VH Feb-Apr <= -18 dB) without
  regrowth (vhdry_trend <= 1 dB), flat (slope <= 3 deg), canopy rise (vhrise >= 1.5 dB). No water signal required.
WorldCover non-rice rules: tree / built = no crop cycle (VH std Mar-Sep <= 1 dB; tree VH mean >= -15.5, built >= -14,
  dark built VH < -14 with VV mean >= -9); water = VH max Mar-Sep <= -21 dB.
Honest comparison: pixels v3 never saw (audit of v3 + candidate labels): v3 = its published map, v4 = out-of-fold,
split edge / interior with the edge flag of data/model/rice_v3/qc_independent_labels.csv.
Outputs: data/model/rice_v4/ (same files as v3) + compare_v3_v4.json, change_v3_v4.tif, dsr_unsure.csv, test_v3_v4.csv.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
from rice_model import worldcover, spaced_sample  # noqa: E402

OUT = PROJ / "data/model/rice_v6"
PREV = PROJ / "data/model/rice_v5"         # the previous model this one is compared with
V4 = PROJ / "data/model/rice_v4"           # its field audit + bins
USER_CANE = [1348570, 988937]              # pixel_id = row * 2219 + col
CHECK = [1348570, 1142172, 988937, 1137502]
AUDITS = [PROJ / f"data/model/rice_v{v}/audit/audit_results.csv" for v in (1, 2, 3, 4)]
LAB = PROJ / "data/labels_v1"
SEED, BLOCK = 0, 50
ACRE = 100 / 4046.856


def metrics(y, p):
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[1, 0], zero_division=0)
    return {"n": int(len(y)), "acc": float((y == p).mean()), "rice_P": pr[0], "rice_R": rc[0], "rice_F1": f[0], "non_F1": f[1]}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grid = json.loads((PROJ / "data/grid/grid.json").read_text())
    with rasterio.open(PROJ / "data/features/s1_features_v4.tif") as r:
        X = r.read(); names = list(r.descriptions); prof = r.profile.copy()
    with rasterio.open(PROJ / "data/ancillary/dem.tif") as r:
        X = np.concatenate([X, r.read().astype("float32")]); names += ["elevation_m", "slope_deg"]
    NB = len(names)                                   # v5 feature set = the first NB bands
    with rasterio.open(PROJ / "data/features/s1_features_v6_extra.tif") as r:
        X = np.concatenate([X, r.read()]); names += list(r.descriptions)
    nx = {n: i for i, n in enumerate(names)}
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    H, W = inside.shape
    print(f"features {X.shape} in {time.time() - t0:.0f} s", flush=True)

    # ---- GCPs (3x3)
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    g = g[g.usable_for_training].copy(); g["y"] = (g.class_2026 == "Rice").astype(int)
    S = [dict(row=r.row + dr, col=r.col + dc, y=r.y, group=r.gcp_id, source="gcp", cls=r.class_2026, sid=r.gcp_id)
         for r in g.itertuples() for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    # ---- audits v1-v3 (high / medium)
    a = pd.concat([pd.read_csv(f).assign(sid=lambda q, v=f.parts[-3]: v + "_" + q.audit_id) for f in AUDITS], ignore_index=True)
    ah = a[a.label.isin(["Rice", "Non-rice"]) & a.confidence.isin(["high", "medium"])]
    S += [dict(row=r.row, col=r.col, y=int(r.label == "Rice"), source="audit_" + r.sid.split("_")[1], cls=f"audit_{r.subtype}",
               sid=r.sid) for r in ah.itertuples()]
    # ---- labels_v1 candidates (high / medium, not Unsure)
    key = pd.read_csv(LAB / "key/candidates_key.csv")
    cl = pd.concat([pd.read_csv(f) for f in sorted((LAB / "labels").glob("batch_*.csv"))]).merge(key, on="audit_id")
    ch = cl[cl.confidence.isin(["high", "medium"]) & (cl.label != "Unsure")]
    S += [dict(row=r.row, col=r.col, y=int(r.label == "Rice"), source="candidates", cls=f"cand_{r.label}",
               sid="cand_" + r.audit_id) for r in ch.itertuples()]
    # ---- user-checked sugarcane (v5 said rice)
    S += [dict(row=p // 2219, col=p % 2219, y=0, source="user", cls="user_sugarcane", sid=f"user_{p}") for p in USER_CANE]
    # ---- Unsure pixels that pass the DSR rule -> rice
    u = pd.concat([a[a.label == "Unsure"][["sid", "row", "col"]],
                   cl[cl.label == "Unsure"].assign(sid=lambda q: "cand_" + q.audit_id)[["sid", "row", "col"]]], ignore_index=True)
    ur, uc = u.row.to_numpy(), u.col.to_numpy()
    tr = ("RO164_DES", "RO099_ASC")
    tm = lambda v, months: np.mean([X[nx[f"{t}_{v}_{m}"]] for t in tr for m in months], 0)   # noqa: E731
    harv = tm("VH", ["202511"]) - tm("VH", ["202601"]); vh_dry = tm("VH", ["202602", "202603", "202604"])
    u["vh_harvest_drop"] = harv[ur, uc]; u["vh_dry"] = vh_dry[ur, uc]
    u["vhdry_trend"] = np.nanmean([X[nx[f"{t}_vhdry_trend"], ur, uc] for t in tr], 0)
    u["vhrise"] = np.nanmean([X[nx[f"{t}_vhrise"], ur, uc] for t in tr], 0)
    u["nwet"] = np.nansum([X[nx[f"{t}_nwet_JunAug"], ur, uc] for t in tr], 0)
    u["slope"] = X[nx["slope_deg"], ur, uc]
    u["dsr"] = ((u.vh_harvest_drop >= 2) & (u.vh_dry <= -18) & (u.vhdry_trend <= 1) & (u.slope <= 3) & (u.vhrise >= 1.5))
    u.round(3).to_csv(OUT / "dsr_unsure.csv", index=False)
    print(f"Unsure {len(u)} -> DSR rice {int(u.dsr.sum())}", flush=True)
    S += [dict(row=r.row, col=r.col, y=1, source="dsr_unsure", cls="dsr_rice", sid=r.sid) for r in u[u.dsr].itertuples()]
    avoid = np.vstack([g[["row", "col"]].to_numpy(), a[["row", "col"]].to_numpy(), cl[["row", "col"]].to_numpy()])
    # ---- WorldCover non-rice, checked against 2026 SAR (no crop cycle / permanent water)
    wc = worldcover(grid)
    vh = np.stack([tm("VH", [f"2026{m:02d}"]) for m in range(3, 10)])
    vh_mean, vh_max, vh_std = vh.mean(0), vh.max(0), vh.std(0); del vh
    vv_mean = tm("VV", [f"2026{m:02d}" for m in range(3, 10)]); stable = vh_std <= 1
    rules = {"tree": ((wc == 10) & stable & (vh_mean >= -15.5), 200), "built": ((wc == 50) & stable & (vh_mean >= -14), 120),
             "built_dark": ((wc == 50) & stable & (vh_mean < -14) & (vv_mean >= -9), 80),
             "water": ((wc == 80) & (vh_max <= -21), 80)}
    for cls, (m, n) in rules.items():
        rr, cc = np.nonzero(m & inside)
        pts = spaced_sample(rr, cc, n, 30, rng, avoid) if len(rr) else np.empty((0, 2))
        print(f"  {cls}: {m.sum()} candidates -> {len(pts)}", flush=True)
        S += [dict(row=int(p[0]), col=int(p[1]), y=0, source="worldcover", cls=cls, sid=f"wc_{cls}_{i}") for i, p in enumerate(pts)]
    del wc, vh_mean, vh_max, vh_std, vv_mean, stable, harv, vh_dry
    S = pd.DataFrame(S)
    S = S[(S.row >= 0) & (S.row < H) & (S.col >= 0) & (S.col < W)]
    # one label per pixel: GCP first, then the newest audit / candidates, DSR last
    order = {"user": -1, "gcp": 0, "audit_v4": 1, "audit_v3": 2, "candidates": 3, "audit_v2": 4, "audit_v1": 5, "dsr_unsure": 6, "worldcover": 7}
    S = S.assign(o=S.source.map(order)).sort_values("o").drop_duplicates(["row", "col"]).drop(columns="o")
    S["group"] = np.where(S.source == "gcp", S.group, "blk" + (S.row // BLOCK).astype(str) + "_" + (S.col // BLOCK).astype(str))
    S = S.reset_index(drop=True)
    F = X[:, S.row, S.col].T
    ok = np.isfinite(F).all(1); S, F = S[ok].reset_index(drop=True), F[ok]
    print(f"samples {len(S)} px (rice {int(S.y.sum())}); by source:", S.groupby(["source", "y"]).size().to_dict(), flush=True)

    # ---- CV (out-of-fold)
    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                  n_jobs=os.cpu_count(), random_state=SEED, eval_metric="logloss")
    spw = (1 - S.y).sum() / max(S.y.sum(), 1)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, S.y, S.group))
    for col, k in [("oof_v5feat", NB), ("oof_prob", F.shape[1])]:
        oof = np.zeros(len(S))
        for trn, te in folds:
            oof[te] = XGBClassifier(**params, scale_pos_weight=spw).fit(F[trn, :k], S.y.iloc[trn]).predict_proba(F[te, :k])[:, 1]
        S[col] = oof
    pv = {"v5feat": "oof_v5feat", "v6": "oof_prob"}
    gcp = S[S.source == "gcp"].groupby("group").agg(y=("y", "first"), **{m: (c, "mean") for m, c in pv.items()})
    cmp = {"gcp_point_cv": {m: metrics(gcp.y.to_numpy(), (gcp[m] >= 0.5).astype(int).to_numpy()) for m in pv}}
    cmp["oof_by_source"] = {s_: {m: metrics(q.y.to_numpy(), (q[c] >= 0.5).astype(int).to_numpy()) for m, c in pv.items()}
                            for s_, q in S.groupby("source")}
    cmp["oof_sugarcane_labels"] = {m: float((S[S.cls.str.contains("ugar")][c] >= 0.5).mean()) for m, c in pv.items()}
    # field audit of v4, area-weighted by v4's probability bins over the whole map (as in v5's report)
    key = pd.read_csv(V4 / "audit/key/audit_key.csv")[["audit_id", "bin"]].assign(sid=lambda q: "rice_v4_" + q.audit_id)
    T = S[S.source == "audit_v4"].merge(key, on="sid")
    p4 = rasterio.open(V4 / "rice_prob.tif").read(1)
    bins = {"rice_high": (90, 100), "rice_med": (70, 89), "rice_low": (50, 69), "non_near": (31, 49), "non_med": (11, 30), "non_low": (0, 10)}
    Wb = {b: ((p4 >= lo) & (p4 <= hi) & inside).sum() * ACRE for b, (lo, hi) in bins.items()}
    aw = {}
    for m, c in pv.items():
        cm_ = np.zeros((2, 2))
        for b, q in T.groupby("bin"):
            pr = (q[c] >= 0.5).astype(int)
            for yy in (0, 1):
                for pp in (0, 1):
                    cm_[yy, pp] += Wb[b] / len(q) * ((q.y == yy) & (pr == pp)).sum()
        aw[m] = {"acc": cm_.trace() / cm_.sum(), "rice_P": cm_[1, 1] / cm_[:, 1].sum(), "rice_R": cm_[1, 1] / cm_[1].sum(),
                 "n": int(len(T))}
    cmp["field_audit_area_weighted_oof"] = aw
    print(json.dumps({k: cmp[k] for k in ("gcp_point_cv", "field_audit_area_weighted_oof", "oof_sugarcane_labels")}, indent=1,
                     default=lambda v: round(float(v), 3)), flush=True)
    print("oof acc by source:", {s_: {m: round(v["acc"], 3) for m, v in d.items()} for s_, d in cmp["oof_by_source"].items()}, flush=True)

    # ---- final model and map
    model = XGBClassifier(**params, scale_pos_weight=spw).fit(F, S.y)
    model.save_model(OUT / "model.json")
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
    flat = X.reshape(len(names), -1)
    valid = inside.ravel().copy()
    for b in range(len(names)):
        valid &= np.isfinite(flat[b])
    prob = np.full(H * W, 255, np.uint8); idx = np.nonzero(valid)[0]
    for s in range(0, len(idx), 500_000):
        b = idx[s:s + 500_000]
        prob[b] = np.round(model.predict_proba(flat[:, b].T)[:, 1] * 100).astype(np.uint8)
    del X, flat
    cmp["check_pixels"] = {str(p): {"v5": int(rasterio.open(PREV / "rice_prob.tif").read(1)[p // W, p % W]), "v6": int(prob[p])}  # prob is still flat
                           for p in CHECK}
    print("check pixels:", cmp["check_pixels"], flush=True)
    prob = prob.reshape(H, W)
    rice = np.where(prob == 255, 255, (prob >= 50).astype(np.uint8)).astype(np.uint8)
    p1 = dict(prof, count=1, dtype="uint8", nodata=255, predictor=2)
    for name, arr, desc in [("rice_prob.tif", prob, "rice probability x100"), ("rice_map.tif", rice, "1 rice, 0 non-rice")]:
        with rasterio.open(OUT / name, "w", **p1) as dst:
            dst.write(arr, 1); dst.set_band_description(1, desc)
    r1 = rasterio.open(PREV / "rice_map.tif").read(1)
    chg = np.full_like(rice, 255)
    m = (rice != 255) & (r1 != 255)
    chg[m & (r1 == 1) & (rice == 1)] = 3; chg[m & (r1 == 0) & (rice == 0)] = 0
    chg[m & (r1 == 1) & (rice == 0)] = 1; chg[m & (r1 == 0) & (rice == 1)] = 2
    with rasterio.open(OUT / "change_v5_v6.tif", "w", **p1) as dst:
        dst.write(chg, 1); dst.set_band_description(1, "0 non both, 1 rice v5 only (removed), 2 rice v6 only (added), 3 rice both")
    area = {"rice_acres_v5": float((r1 == 1).sum() * ACRE), "rice_acres_v6": float((rice == 1).sum() * ACRE),
            "removed_acres": float((chg == 1).sum() * ACRE), "added_acres": float((chg == 2).sum() * ACRE)}
    # cleaned map as v3: 3x3 mean probability, then rice patches / non-rice holes smaller than 10 px (0.25 acre) removed
    from scipy import ndimage
    pf = np.where(prob == 255, np.nan, prob.astype("float32"))
    num = ndimage.uniform_filter(np.nan_to_num(pf), 3); den = ndimage.uniform_filter(np.isfinite(pf).astype("float32"), 3)
    with np.errstate(all="ignore"):
        ps = num / den
    clean = (ps >= 50) & (prob != 255)
    lab, n = ndimage.label(clean); sz = np.bincount(lab.ravel()); clean[(sz[lab] < 10) & (lab > 0)] = False
    lab, n = ndimage.label(~clean & (prob != 255)); sz = np.bincount(lab.ravel()); clean[(sz[lab] < 10) & (lab > 0)] = True
    cm = np.where(prob == 255, 255, clean.astype(np.uint8)).astype(np.uint8)
    with rasterio.open(OUT / "rice_map_clean.tif", "w", **p1) as dst:
        dst.write(cm, 1); dst.set_band_description(1, "1 rice, 0 non-rice (3x3 smoothed, min patch 10 px)")
    c3 = rasterio.open(PREV / "rice_map_clean.tif").read(1)
    area["rice_acres_v5_clean"] = float((c3 == 1).sum() * ACRE); area["rice_acres_v6_clean"] = float((cm == 1).sum() * ACRE)
    cmp["area"] = area
    (OUT / "compare_v5_v6.json").write_text(json.dumps(cmp, indent=1, default=float))
    print("area:", {k: round(v) for k, v in area.items()}, flush=True)
    allg = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    allg["map_prob"] = prob[allg.row, allg.col]
    allg[["gcp_id", "crop", "class_2026", "final_verdict", "map_prob", "lon", "lat"]].to_csv(OUT / "gcp_predictions.csv", index=False)
    import geopandas as gpd
    sx = grid["x0"] + (S.col + 0.5) * 10; sy = grid["y0"] - (S.row + 0.5) * 10
    gpd.GeoDataFrame(S, geometry=gpd.points_from_xy(sx, sy), crs=grid["crs"]).to_file(OUT / "samples.gpkg", driver="GPKG")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    ax[0].imshow(np.ma.masked_equal(rice, 255), cmap="Greens", vmin=0, vmax=1.3)
    ax[0].set_title(f"rice v6: {area['rice_acres_v6']:,.0f} acres, cleaned {area['rice_acres_v6_clean']:,.0f} (v5 {area['rice_acres_v5']:,.0f})")
    ax[1].imshow(np.ma.masked_equal(chg, 255), cmap=ListedColormap(["#f0f0f0", "#d7301f", "#2b8cbe", "#74c476"]), vmin=0, vmax=3)
    ax[1].set_title(f"change v5->v6: red = removed ({area['removed_acres']:,.0f} ac), blue = added ({area['added_acres']:,.0f} ac), green = rice in both")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "overview.png", dpi=80); plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s; top features:", imp.head(10).round(3).to_dict())


if __name__ == "__main__":
    main()

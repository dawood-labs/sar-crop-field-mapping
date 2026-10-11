"""Rice vs non-rice, model v2 (Sentinel-1 + terrain). Changes from v1, all from the blind audit of v1:
  1. features Nov 2025 - Sep 2026 (data/features/s1_monthly_w5_v2.tif, 66 bands): sugarcane / cassava that stay green
     in Dec-Feb were mapped as rice when the model only saw Mar-Sep;
  2. elevation and slope (data/ancillary/dem.tif) as features: no audited pixel above ~8 deg slope was rice;
  3. extra training samples: the audit pixels the analysts labelled with high or medium confidence (1 pixel each), and
     dark built-up / bare pixels (WorldCover built with low NDVI and VH below -14 dB, like A130 at a village edge).
Everything else as v1 (XGBoost, same GCP and WorldCover rules, 5-fold StratifiedGroupKFold).
Because audit pixels are now used for training, the audit is no longer an independent test: v2's agreement with the
analysts is reported OUT-OF-FOLD (each audit pixel predicted by a model that did not see it), next to v1's honest
agreement on the same pixels. A fresh independent audit is needed before trusting v2's numbers fully.
Outputs: data/model/rice_v2/ (same files as v1) + compare_v1_v2.json and change map (v1 -> v2).
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
from rice_model import worldcover, dry_ndvi, spaced_sample  # noqa: E402

OUT = PROJ / "data/model/rice_v2"
V1 = PROJ / "data/model/rice_v1"
SEED = 0


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grid = json.loads((PROJ / "data/grid/grid.json").read_text())
    with rasterio.open(PROJ / "data/features/s1_monthly_w5_v2.tif") as r:
        X = r.read(); names = list(r.descriptions); prof = r.profile.copy()
    with rasterio.open(PROJ / "data/ancillary/dem.tif") as r:
        X = np.concatenate([X, r.read().astype("float32")]); names += ["elevation_m", "slope_deg"]
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    H, W = inside.shape
    print(f"features {X.shape} in {time.time() - t0:.0f} s", flush=True)

    # ---- samples: GCPs (3x3), as v1
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    g = g[g.usable_for_training].copy(); g["y"] = (g.class_2026 == "Rice").astype(int)
    S = [dict(row=r.row + dr, col=r.col + dc, y=r.y, group=r.gcp_id, source="gcp", cls=r.class_2026)
         for r in g.itertuples() for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    # ---- audit pixels (high / medium confidence only)
    a = pd.read_csv(V1 / "audit/audit_results.csv")
    a = a[a.label.isin(["Rice", "Non-rice"]) & a.confidence.isin(["high", "medium"])]
    S += [dict(row=r.row, col=r.col, y=int(r.label == "Rice"), group=r.audit_id, source="audit", cls=f"audit_{r.subtype}")
          for r in a.itertuples()]
    avoid = np.vstack([g[["row", "col"]].to_numpy(), a[["row", "col"]].to_numpy()])
    # ---- WorldCover non-rice, checked against 2026 data (v1 rules + dark built / bare)
    wc = worldcover(grid); dn = dry_ndvi()
    vh = X[[i for i, n in enumerate(names) if "_VH_" in n and n.split("_")[-1] >= "202603"]]
    vh_mean, vh_max = np.nanmean(vh, 0), np.nanmax(vh, 0)
    rules = {"tree": ((wc == 10) & (dn >= 0.55), 200), "built": ((wc == 50) & (vh_mean >= -14) & (dn < 0.35), 120),
             "built_dark": ((wc == 50) & (vh_mean < -14) & (dn < 0.25), 80),
             "water": ((wc == 80) & (vh_max <= -21) & (dn < 0.1), 80)}
    for cls, (m, n) in rules.items():
        rr, cc = np.nonzero(m & inside)
        pts = spaced_sample(rr, cc, n, 30, rng, avoid) if len(rr) else np.empty((0, 2))
        print(f"  {cls}: {m.sum()} candidates -> {len(pts)}", flush=True)
        S += [dict(row=int(p[0]), col=int(p[1]), y=0, group=f"blk{p[0] // 200}_{p[1] // 200}", source="worldcover", cls=cls) for p in pts]
    S = pd.DataFrame(S)
    S = S[(S.row >= 0) & (S.row < H) & (S.col >= 0) & (S.col < W)].reset_index(drop=True)
    F = X[:, S.row, S.col].T
    ok = np.isfinite(F).all(1); S, F = S[ok].reset_index(drop=True), F[ok]
    print(f"samples {len(S)} px (rice {int(S.y.sum())}); by source:", S.groupby(["source", "y"]).size().to_dict(), flush=True)

    # ---- CV
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import precision_recall_fscore_support
    from xgboost import XGBClassifier
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                  n_jobs=-1, random_state=SEED, eval_metric="logloss")
    spw = (1 - S.y).sum() / max(S.y.sum(), 1)
    oof = np.zeros(len(S))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, S.y, S.group):
        oof[te] = XGBClassifier(**params, scale_pos_weight=spw).fit(F[tr], S.y.iloc[tr]).predict_proba(F[te])[:, 1]
    S["oof_prob"] = oof
    gcp = S[S.source == "gcp"].groupby("group").agg(y=("y", "first"), prob=("oof_prob", "mean"), cls=("cls", "first"))
    gp, gr, gf, _ = precision_recall_fscore_support(gcp.y, (gcp.prob >= 0.5).astype(int), labels=[1, 0], zero_division=0)
    au = S[S.source == "audit"].merge(a[["audit_id", "model_prob", "bin"]], left_on="group", right_on="audit_id")
    au["v1_pred"] = (au.model_prob >= 50).astype(int); au["v2_pred"] = (au.oof_prob >= 0.5).astype(int)
    v1_agree, v2_agree = float((au.v1_pred == au.y).mean()), float((au.v2_pred == au.y).mean())
    per_bin = au.groupby("bin").apply(lambda q: pd.Series({"n": len(q), "v1_agree": (q.v1_pred == q.y).mean(),
                                                          "v2_agree_oof": (q.v2_pred == q.y).mean()}), include_groups=False)
    v1m = json.loads((V1 / "cv_metrics.json").read_text())
    cmp = {"gcp_point_cv": {"v1": v1m["gcp_point"], "v2": {"rice_precision": gp[0], "rice_recall": gr[0], "rice_f1": gf[0], "nonrice_f1": gf[1]}},
           "audit_agreement_high_medium": {"n": int(len(au)), "v1_honest": v1_agree, "v2_out_of_fold": v2_agree,
                                           "per_bin": per_bin.round(3).reset_index().to_dict("records")}}
    print(json.dumps(cmp, indent=1, default=float), flush=True)
    wrong = gcp[gcp.y != (gcp.prob >= 0.5)]
    print("GCPs still wrong (oof):", wrong.round(2).to_dict("index"), flush=True)

    # ---- final model and map
    model = XGBClassifier(**params, scale_pos_weight=spw).fit(F, S.y)
    model.save_model(OUT / "model.json")
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
    flat = X.reshape(len(names), -1).T
    valid = inside.ravel() & np.isfinite(flat).all(1)
    prob = np.full(H * W, 255, np.uint8); idx = np.nonzero(valid)[0]
    for s in range(0, len(idx), 1_000_000):
        b = idx[s:s + 1_000_000]
        prob[b] = np.round(model.predict_proba(flat[b])[:, 1] * 100).astype(np.uint8)
    prob = prob.reshape(H, W)
    rice = np.where(prob == 255, 255, (prob >= 50).astype(np.uint8)).astype(np.uint8)
    p1 = dict(prof, count=1, dtype="uint8", nodata=255, predictor=2)
    for name, arr, desc in [("rice_prob.tif", prob, "rice probability x100"), ("rice_map.tif", rice, "1 rice, 0 non-rice")]:
        with rasterio.open(OUT / name, "w", **p1) as dst:
            dst.write(arr, 1); dst.set_band_description(1, desc)
    r1 = rasterio.open(V1 / "rice_map.tif").read(1)
    acre = 100 / 4046.856
    chg = np.full_like(rice, 255)
    m = (rice != 255) & (r1 != 255)
    chg[m & (r1 == 1) & (rice == 1)] = 3; chg[m & (r1 == 0) & (rice == 0)] = 0
    chg[m & (r1 == 1) & (rice == 0)] = 1; chg[m & (r1 == 0) & (rice == 1)] = 2
    with rasterio.open(OUT / "change_v1_v2.tif", "w", **p1) as dst:
        dst.write(chg, 1); dst.set_band_description(1, "0 non both, 1 rice v1 only (removed), 2 rice v2 only (added), 3 rice both")
    area = {"rice_acres_v1": float((r1 == 1).sum() * acre), "rice_acres_v2": float((rice == 1).sum() * acre),
            "removed_acres": float((chg == 1).sum() * acre), "added_acres": float((chg == 2).sum() * acre),
            "audit_based_rice_estimate_v1_acres": 29254}
    cmp["area"] = area
    (OUT / "compare_v1_v2.json").write_text(json.dumps(cmp, indent=1, default=float))
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
    ax[0].set_title(f"rice v2: {area['rice_acres_v2']:,.0f} acres (v1 {area['rice_acres_v1']:,.0f})")
    ax[1].imshow(np.ma.masked_equal(chg, 255), cmap=ListedColormap(["#f0f0f0", "#d7301f", "#2b8cbe", "#74c476"]), vmin=0, vmax=3)
    ax[1].set_title(f"change: red = removed by v2 ({area['removed_acres']:,.0f} ac), blue = added ({area['added_acres']:,.0f} ac), green = rice in both")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "overview.png", dpi=80); plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s; top features:", imp.head(10).round(3).to_dict())


if __name__ == "__main__":
    main()

"""Rice vs non-rice from Sentinel-1 only (features: data/features/s1_monthly_w5.tif, Mar-Sep 2026).

Training samples
  - verified GCPs (data/gcps/eda/gcp_verdicts.csv, usable_for_training): Rice = 1, everything else = 0; each point
    contributes its 3 x 3 pixels (same field), grouped by point for cross-validation.
  - extra NON-RICE samples from ESA WorldCover 2021 (tree cover, built-up, permanent water), each checked against our
    own 2026 data so a stale WorldCover pixel is not used:
      tree   : dry-season (Jan-Apr 2026) median cloud-free NDVI >= 0.55
      built  : mean VH >= -14 dB and dry-season NDVI < 0.35
      water  : mean VH <= -21 dB in every month and dry-season NDVI < 0.1
    Sentinel-2 is used ONLY to filter these samples, never as a model feature. One pixel per sample, at least 300 m
    apart, at least 300 m from any GCP; grouped in 2 km blocks for cross-validation.
Model: XGBoost (depth 4, 400 trees, lr 0.05), 5-fold StratifiedGroupKFold; out-of-fold metrics reported per GCP
(majority of its 3 x 3 pixels) and per pixel. Final model on all samples -> full AOI.
Outputs (data/model/rice_v1/): rice_prob.tif (0-100, 255 outside AOI), rice_map.tif (1 rice, 0 non-rice, 255 outside),
samples.gpkg, cv_metrics.json, gcp_predictions.csv, feature_importance.csv, overview.png, model.json
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
OUT = PROJ / "data/model/rice_v1"
sys.path.insert(0, str(PROJ / "scripts"))
SEED = 0


def worldcover(grid):
    """ESA WorldCover v200 (2021) on the project grid (uint8 classes), cached."""
    f = PROJ / "data/ancillary/worldcover_2021.tif"
    if f.exists():
        return rasterio.open(f).read(1)
    import download as dl, ee
    dl.init_ee(dl.CFG, high_volume=True)
    a = dl.fetch(ee.ImageCollection("ESA/WorldCover/v200").first().select("Map"), ["Map"], np.uint8, grid, 0)
    dl.write_tif(f, a, grid, ["worldcover_2021"], 0, {"source": "ESA/WorldCover/v200 (2021), nearest to 10 m"})
    return a[0]


def dry_ndvi():
    """Median cloud-free NDVI over Jan-Apr 2026 (for sample filtering only)."""
    st = PROJ / "data/stacks"
    with rasterio.open(st / "S2_NDVI.tif") as r:
        d = pd.to_datetime(list(r.descriptions))
        k = [i + 1 for i, x in enumerate(d) if pd.Timestamp("2026-01-01") <= x <= pd.Timestamp("2026-04-30")]
        nd = r.read(k).astype("float32") / 1e4
    with rasterio.open(st / "S2_CLEAR.tif") as r:
        cl = r.read(k) == 1
    nd[~cl] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(nd, axis=0)


def spaced_sample(rows, cols, n, min_px, rng, avoid_rc):
    """Random pixels at least min_px apart from each other and from avoid_rc."""
    from scipy.spatial import cKDTree
    order = rng.permutation(len(rows))
    taken, tree_av = [], cKDTree(avoid_rc) if len(avoid_rc) else None
    pts = []
    for i in order:
        p = (rows[i], cols[i])
        if tree_av is not None and tree_av.query(p)[0] < min_px:
            continue
        if pts and cKDTree(pts).query(p)[0] < min_px:
            continue
        pts.append(p)
        if len(pts) >= n:
            break
    return np.array(pts)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grid = json.loads((PROJ / "data/grid/grid.json").read_text())
    with rasterio.open(PROJ / "data/features/s1_monthly_w5.tif") as r:
        X = r.read()                                        # (n_feat, H, W) float32
        names = list(r.descriptions); prof = r.profile.copy()
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    H, W = inside.shape
    print(f"features {X.shape}, read in {time.time() - t0:.0f} s", flush=True)

    # ---- GCP samples (3x3 per point)
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    g = g[g.usable_for_training].copy()
    g["y"] = (g.class_2026 == "Rice").astype(int)
    S = []
    for r in g.itertuples():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                S.append(dict(row=r.row + dr, col=r.col + dc, y=r.y, group=r.gcp_id, source="gcp", cls=r.class_2026))
    gcp_rc = g[["row", "col"]].to_numpy()

    # ---- extra non-rice from WorldCover, checked against 2026 data
    wc = worldcover(grid)
    dn = dry_ndvi()
    vh = X[[i for i, n in enumerate(names) if "_VH_" in n]]
    vh_mean, vh_max = np.nanmean(vh, 0), np.nanmax(vh, 0)
    rules = {"tree": (wc == 10) & (dn >= 0.55),
             "built": (wc == 50) & (vh_mean >= -14) & (dn < 0.35),
             "water": (wc == 80) & (vh_max <= -21) & (dn < 0.1)}
    want = {"tree": 200, "built": 120, "water": 80}
    for cls, m in rules.items():
        rr, cc = np.nonzero(m & inside)
        pts = spaced_sample(rr, cc, want[cls], 30, rng, gcp_rc) if len(rr) else np.empty((0, 2))
        print(f"  {cls}: {m.sum()} candidate px -> {len(pts)} samples", flush=True)
        for p in pts:
            S.append(dict(row=int(p[0]), col=int(p[1]), y=0, group=f"blk{p[0] // 200}_{p[1] // 200}", source="worldcover", cls=cls))
    S = pd.DataFrame(S)
    S = S[(S.row >= 0) & (S.row < H) & (S.col >= 0) & (S.col < W)].reset_index(drop=True)
    F = X[:, S.row, S.col].T
    ok = np.isfinite(F).all(1)
    S, F = S[ok].reset_index(drop=True), F[ok]
    print(f"samples: {len(S)} px  (rice {int(S.y.sum())}, non-rice {int((1 - S.y).sum())}); by class:",
          S.groupby('cls').size().to_dict(), flush=True)

    # ---- spatial cross-validation
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    from xgboost import XGBClassifier
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                  n_jobs=-1, random_state=SEED, eval_metric="logloss")
    spw = (1 - S.y).sum() / max(S.y.sum(), 1)
    oof = np.zeros(len(S))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, S.y, S.group):
        m = XGBClassifier(**params, scale_pos_weight=spw).fit(F[tr], S.y.iloc[tr])
        oof[te] = m.predict_proba(F[te])[:, 1]
    S["oof_prob"] = oof
    px = S.y.values, (oof >= 0.5).astype(int)
    p, r_, f1, _ = precision_recall_fscore_support(*px, labels=[1, 0], zero_division=0)
    gcp = S[S.source == "gcp"].groupby("group").agg(y=("y", "first"), prob=("oof_prob", "mean"), cls=("cls", "first"))
    gcp["pred"] = (gcp.prob >= 0.5).astype(int)
    gp, gr, gf, _ = precision_recall_fscore_support(gcp.y, gcp.pred, labels=[1, 0], zero_division=0)
    cm = confusion_matrix(gcp.y, gcp.pred, labels=[1, 0]).tolist()
    metrics = {"pixel": {"rice_precision": p[0], "rice_recall": r_[0], "rice_f1": f1[0], "nonrice_f1": f1[1]},
               "gcp_point": {"rice_precision": gp[0], "rice_recall": gr[0], "rice_f1": gf[0], "nonrice_f1": gf[1],
                             "confusion_[rice,non]_rows_true": cm, "n_points": int(len(gcp))}}
    metrics = json.loads(json.dumps(metrics, default=float))
    (OUT / "cv_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("CV:", json.dumps(metrics), flush=True)
    wrong = gcp[gcp.y != gcp.pred]
    print("misclassified GCPs (out-of-fold):", wrong.round(2).to_dict("index"), flush=True)

    # ---- final model, full AOI
    model = XGBClassifier(**params, scale_pos_weight=spw).fit(F, S.y)
    model.save_model(OUT / "model.json")
    imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", header=["importance"])
    flat = X.reshape(len(names), -1).T
    valid = inside.ravel() & np.isfinite(flat).all(1)
    prob = np.full(H * W, 255, np.uint8)
    idx = np.nonzero(valid)[0]
    for s in range(0, len(idx), 1_000_000):
        b = idx[s:s + 1_000_000]
        prob[b] = np.round(model.predict_proba(flat[b])[:, 1] * 100).astype(np.uint8)
    prob = prob.reshape(H, W)
    rice = np.where(prob == 255, 255, (prob >= 50).astype(np.uint8)).astype(np.uint8)
    p1 = dict(prof, count=1, dtype="uint8", nodata=255, predictor=2)
    for name, arr, desc in [("rice_prob.tif", prob, "rice probability x100"), ("rice_map.tif", rice, "1 rice, 0 non-rice")]:
        with rasterio.open(OUT / name, "w", **p1) as dst:
            dst.write(arr, 1); dst.set_band_description(1, desc)
    acre = 100 / 4046.856
    area = {"aoi_acres": float(inside.sum() * acre), "rice_acres": float((rice == 1).sum() * acre),
            "nonrice_acres": float((rice == 0).sum() * acre)}
    area["rice_share"] = area["rice_acres"] / area["aoi_acres"]
    (OUT / "area.json").write_text(json.dumps(area, indent=1))
    print("area:", {k: round(v, 3) for k, v in area.items()}, flush=True)

    # ---- per-GCP prediction table (all 100 points, incl. not used / Cant Access)
    allg = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    allg["map_prob"] = prob[allg.row, allg.col]
    allg["map_class"] = np.where(allg.map_prob >= 50, "Rice", "Non-rice")
    allg = allg.merge(gcp[["prob"]].rename(columns={"prob": "cv_prob"}), left_on="gcp_id", right_index=True, how="left")
    allg[["gcp_id", "crop", "class_2026", "final_verdict", "usable_for_training", "cv_prob", "map_prob", "map_class", "lon", "lat"]] \
        .to_csv(OUT / "gcp_predictions.csv", index=False)
    import geopandas as gpd
    sx = grid["x0"] + (S.col + 0.5) * 10; sy = grid["y0"] - (S.row + 0.5) * 10
    gpd.GeoDataFrame(S, geometry=gpd.points_from_xy(sx, sy), crs=grid["crs"]).to_file(OUT / "samples.gpkg", driver="GPKG")

    # ---- overview figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    show = np.ma.masked_equal(prob, 255)
    im = ax[0].imshow(show, cmap="RdYlGn_r" if False else "YlGn", vmin=0, vmax=100); plt.colorbar(im, ax=ax[0], shrink=0.7, label="rice probability %")
    ax[1].imshow(np.ma.masked_equal(rice, 255), cmap="Greens", vmin=0, vmax=1.3)
    for a in ax:
        a.scatter(allg.col, allg.row, c=np.where(allg.class_2026 == "Rice", "blue", np.where(allg.class_2026.isin(["Exclude", "No label"]), "gray", "red")),
                  s=14, edgecolors="white", linewidths=0.5)
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_title("rice probability (S1 Mar-Sep 2026)"); ax[1].set_title(f"rice map: {area['rice_acres']:,.0f} acres ({area['rice_share']:.0%} of AOI)")
    fig.suptitle("points: blue = rice GCP, red = non-rice GCP, gray = excluded / no label"); fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=80); plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s; top features:", imp.head(8).round(3).to_dict())


if __name__ == "__main__":
    main()

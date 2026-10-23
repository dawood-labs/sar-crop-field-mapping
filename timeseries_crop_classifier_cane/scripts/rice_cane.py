"""Rice INSIDE the sugarcane class (user, 16 Oct 2026: "just cane chahiye" - take rice out of layer2_final class 1; rice/cassava
mix-up elsewhere is acceptable). Built from scratch on per-pixel RELATIVE, consistency-based signals:
  * data/features/s1_runs_2026.tif   (per-date S1, both tracks merged: flood runs, VV-VH double-bounce runs, smoothed extremes,
                                      May green-up; see scripts/s1_runs.py)
  * data/features/s1_15d_2026nov.tif (15-day VH / VV-VH per track, Apr-Oct, each minus the pixel's own median = anomaly series)
  * Sep-Oct 5-3-2 colour trajectory  (scripts/rice_stage2.trajectory: yellowness / greenness / hue first, last, slope, drop)
Labels: rice = user rice polygons (Site39, Site27 not-cane, gcps rice) + reviewer rice blocks (sites/layer2_labels_all.csv);
cane = user cane polygons (Site39, Site27, site25, Site33, Site41) + reviewer sugarcane blocks. Test-set blocks excluded.
Train: LightGBM, leave-one-district-out report (rice recall, false rice on cane) -> sites/layer2/rice_cane_model.joblib.
Apply: p(rice) on every class-1 pixel -> <site>/data/model/rice_cane/: rice_prob_2026.tif, rice_2026.tif (1 rice, 0 cane, 255 out),
class4_2026.tif (1 sugarcane, 7 rice, 281 cassava, 0 none), rice.json (acres).
usage: python scripts/rice_cane.py feat | train | apply [site..]   (feat = cache label features, run several in parallel)    env THR=0.6 SMOOTH=5"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd, joblib, lightgbm as lgb
from rasterio.features import rasterize
from scipy import ndimage
import site_paths as SP
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q; from rice_stage2 import trajectory, COLS as CCOLS  # noqa
P = Path(__file__).resolve().parents[1]; ACRE = 0.0247105; THR = float(os.environ.get("THR", 0.6)); SMOOTH = int(os.environ.get("SMOOTH", 5))
U = P / "sites/user_labels"; SITE_DIR = {"site25": "site25_2026", "Site41": "site41_2026", "Site33": "site33_2026"}
HM = [f"2026{m:02d}{h}" for m in range(4, 11) for h in "ab"][:-1]            # 202604a .. 202610a
PARAMS = dict(n_estimators=400, num_leaves=15, min_child_samples=50, learning_rate=0.05, class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0, n_jobs=8)


def user_polys():
    rd = lambda f: gpd.read_file(U / f).to_crs(32648)
    parts = [rd("site39_rice_gcps_2026-10-16.gpkg").assign(site="site39_2026", y=1), rd("site39_cane_gcps_2026-10-16.gpkg").assign(site="site39_2026", y=0),
             rd("cane_polygons_site25_2026-10-15.gpkg").assign(site="site25_2026", y=0)]
    nb = rd("site27_gcps_2026-10-16.gpkg"); parts.append(nb.assign(site="bench_site27_2026", y=(nb.attribute == "not-cane").astype(int)))
    c = rd("cane_polygons_2026-10-15.gpkg"); parts.append(c.assign(site=c.site.map(SITE_DIR), y=0))
    g = rd("gcps_country_2026-10-15_analysis.gpkg"); g = g[g.call2.str.startswith("rice")]; parts.append(g.assign(site=g.site.map(SITE_DIR), y=1))
    return pd.concat([p[["site", "y", "geometry"]] for p in parts], ignore_index=True)


def features(site, mask):
    """DataFrame of features for pixels in mask (row-major order of np.nonzero(mask))."""
    S = SP.site_dir(site) / "data"; rr, cc = np.nonzero(mask); f = {}
    with rasterio.open(S / "features/s1_runs_2026.tif") as r:
        for i, n in enumerate(r.descriptions, 1):
            a = r.read(i)[rr, cc].astype("float32"); f[n] = np.where(a == -32768, np.nan, a / (100 if n in ("gap_sm_max", "vv_sm_max", "vh_sm_min", "vh_rise_may") else 1))
    with rasterio.open(S / "features/s1_15d_2026nov.tif") as r:
        names = list(r.descriptions)
        for orb in ("RO164_DES", "RO099_ASC"):
            vh = np.stack([r.read(names.index(f"{orb}_VH_{p}") + 1)[rr, cc].astype("float32") for p in HM]); vv = np.stack([r.read(names.index(f"{orb}_VV_{p}") + 1)[rr, cc].astype("float32") for p in HM])
            vh[vh <= -32000] = np.nan; vv[vv <= -32000] = np.nan; vh /= 100; vv /= 100; gap = vv - vh
            avh = vh - np.nanmedian(vh, 0); agap = gap - np.nanmedian(gap, 0)
            for k, p in enumerate(HM): f[f"{orb[:5]}_aVH_{p}"] = avh[k]; f[f"{orb[:5]}_gap_{p}"] = gap[k]
    col = trajectory(site, mask)
    for k in CCOLS: f["c_" + k] = col[k][rr, cc]
    return pd.DataFrame(f)


def labels(site):
    S = SP.site_dir(site) / "data"
    with rasterio.open(S / "grid/index.tif") as r: shp, tr = r.shape, r.transform
    lab = np.full(shp, -1, "int8"); src = np.zeros(shp, "uint8")
    rev = pd.read_csv(P / "sites/layer2_labels_all.csv"); rev = rev[(rev.site == site) & (rev.confidence != "low")]
    for _, q in rev.iterrows():
        y = 1 if (q.answer == "not_cane" and q.what == "rice") else (0 if (q.answer == "cane" and q.what == "sugarcane") else -1)
        if y >= 0: lab[q.row - 1:q.row + 2, q.col - 1:q.col + 2] = y; src[q.row - 1:q.row + 2, q.col - 1:q.col + 2] = 2
    up = user_polys(); up = up[up.site == site]
    for _, q in up.iterrows():
        m = rasterize([(q.geometry, 1)], out_shape=shp, transform=tr, dtype="uint8").astype(bool); lab[m] = q.y; src[m] = 1
    key = pd.read_csv(P / "sites/testset_v1/key/key.csv"); key = key[key.site == site]
    for _, q in key.iterrows(): lab[q.row - 2:q.row + 3, q.col - 2:q.col + 3] = -1          # never train on test-set blocks
    return lab, src


def label_feats(s):
    """Cached label-pixel feature table for one district -> sites/<s>/data/features/rice_cane_labels.parquet (None if no labels)."""
    f = SP.site_dir(s) / "data/features/rice_cane_labels.parquet"
    if f.exists(): return pd.read_parquet(f)
    lab, src = labels(s); m = lab >= 0
    if not m.any(): return None
    X = features(s, m); X["y"] = lab[m]; X["src"] = src[m]; X["site"] = s; X.to_parquet(f)
    print(f"{s}: rice {int((X.y == 1).sum())} px (user {int(((X.y == 1) & (X.src == 1)).sum())}), cane {int((X.y == 0).sum())} px (user {int(((X.y == 0) & (X.src == 1)).sum())})", flush=True)
    return X


def weights(D, cap=1000):
    """Per-pixel weights so that no (site, class) group counts more than `cap` pixels (Site27's 6,800 user rice px would dominate)."""
    n = D.groupby(["site", "y"]).y.transform("size"); return np.minimum(1.0, cap / n).values


def train(sites):
    rows = [x for x in (label_feats(s) for s in sites) if x is not None]
    D = pd.concat(rows, ignore_index=True); D.to_parquet(P / "sites/layer2/rice_cane_train.parquet"); cols = [c for c in D.columns if c not in ("y", "src", "site")]
    print("\n=== leave-one-district-out (pixel level) ===")
    out = []
    for s in sorted(D.site.unique()):
        tr, te = D[D.site != s], D[D.site == s]
        m = lgb.LGBMClassifier(**PARAMS).fit(tr[cols], tr.y, sample_weight=weights(tr)); p = m.predict_proba(te[cols])[:, 1]
        for thr in (0.5, 0.6, 0.7):
            r = te.y == 1; c = te.y == 0
            out.append(dict(site=s, thr=thr, rice_px=int(r.sum()), rice_recall=round(float((p[r] >= thr).mean()), 2) if r.any() else np.nan,
                            cane_px=int(c.sum()), false_rice=round(float((p[c] >= thr).mean()), 3) if c.any() else np.nan,
                            user_rice_recall=round(float((p[r & (te.src == 1)] >= thr).mean()), 2) if (r & (te.src == 1)).any() else np.nan,
                            user_cane_false=round(float((p[c & (te.src == 1)] >= thr).mean()), 3) if (c & (te.src == 1)).any() else np.nan))
    R = pd.DataFrame(out); pd.set_option("display.width", 250); print(R[R.thr == THR].to_string(index=False))
    for thr in (0.5, 0.6, 0.7):
        q = R[R.thr == thr]; print(f"thr {thr}: pooled rice recall {np.average(q.rice_recall.fillna(0), weights=q.rice_px):.2f}, false rice on cane {np.average(q.false_rice.fillna(0), weights=q.cane_px):.3f}")
    R.to_csv(P / "sites/layer2/rice_cane_lodo.csv", index=False)
    m = lgb.LGBMClassifier(**PARAMS).fit(D[cols], D.y, sample_weight=weights(D)); joblib.dump({"model": m, "cols": cols}, P / "sites/layer2/rice_cane_model.joblib")
    imp = pd.Series(m.feature_importances_, cols).sort_values(ascending=False); print("\ntop features:", imp.head(15).to_dict())


def apply(sites):
    mdl = joblib.load(P / "sites/layer2/rice_cane_model.joblib"); m, cols = mdl["model"], mdl["cols"]
    for s in sites:
        S = SP.site_dir(s) / "data"; o = S / "model/rice_cane"; o.mkdir(parents=True, exist_ok=True)
        with rasterio.open(S / "model/layer2_final/class3_2026.tif") as r: cls, prof = r.read(1), r.profile.copy()
        mask = cls == 1; X = features(s, mask); p = np.full(cls.shape, np.nan, "float32"); p[mask] = m.predict_proba(X[cols])[:, 1]
        if SMOOTH > 1:                                                     # mean of p over the class-1 neighbourhood (field-level consistency)
            num = ndimage.uniform_filter(np.nan_to_num(p), SMOOTH); den = ndimage.uniform_filter(mask.astype("float32"), SMOOTH)
            with np.errstate(all="ignore"): ps = np.where(mask, num / den, np.nan)
        else: ps = p
        rice = mask & (ps >= THR)
        pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
        with rasterio.open(o / "rice_prob_2026.tif", "w", **pr) as d: d.write(np.where(mask, np.round(ps * 100), 255).astype("uint8"), 1)
        with rasterio.open(o / "rice_2026.tif", "w", **pr) as d: d.write(np.where(cls == 65535, 255, np.where(rice, 1, 0)).astype("uint8"), 1)
        c4 = np.where(cls == 65535, 65535, np.where(rice, 7, cls)).astype("uint16")
        with rasterio.open(o / "class4_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(c4, 1)
        Q.pseudocolor(o / "rice_prob_2026.tif", Q.PROB_STOPS); Q.paletted(o / "rice_2026.tif", {0: ("#33ef2a", "sugarcane"), 1: ("#1B5E20", "rice (taken out of sugarcane)")})
        Q.paletted(o / "class4_2026.tif", {0: ("#e6e6e6", "none"), 1: ("#33ef2a", "sugarcane"), 7: ("#1B5E20", "rice"), 281: ("#8b1a1a", "cassava")})
        res = {"site": s, "thr": THR, "smooth": SMOOTH, "sugarcane_class_ac": round(float(mask.sum() * ACRE)), "rice_ac": round(float(rice.sum() * ACRE)), "sugarcane_final_ac": round(float((c4 == 1).sum() * ACRE))}
        (o / "rice.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)


if __name__ == "__main__":
    ALL = sorted(p.name for p in SP.all_sites("*_2026") if (p / "data/model/layer2_final").exists())
    sites = sys.argv[2:] or ALL
    if sys.argv[1] == "feat": [label_feats(s) for s in sites]
    else: train(sites) if sys.argv[1] == "train" else apply(sites)
    print("RICE_CANE_DONE")

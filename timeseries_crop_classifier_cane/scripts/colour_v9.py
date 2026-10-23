"""Static colour model v9: CANE vs NON-CANE on one late-season image, with RELATIVE colour (user, 21 Oct 2026).
Features (11): B5, B3, B2, B5/B3, B3/B2, hue, sat, val (colour_multi.feats) + B4, B5/B4, B4/B3.
Normalisation: per image (site + date), z = (x - median) / IQR of the image's SURE-CANE pixels (layer-1 prob >= 90 and
layer2_final sugarcane, clear). So every pixel is judged against the cane of its own image, like a QGIS stretch.
Training: sites/layer2/colour_separability_v2_b4/samples.parquet (277 user polygons, every clear date >= 10 Sep, 103,863 px)
with sites/layer2/colour_image_stats_surecane.pkl; polygon-level CV 0.993, leave-one-site-out 0.989.
Apply: the district's 3 clearest late dates (colour_multi.best_dates), per pixel the latest clear one; p(cane) on the layer-1
cane-like mask; SMOOTH (9x9 mean inside the mask); class 1 sugarcane where p >= THR else 304 non-cane; 65535 cloud/unknown.
Output <site>/data/model/<OUT_TAG default colour_v9>/: cane_colour_prob_2026.tif (p cane 0-100), class_2026.tif, colour.json
usage: python scripts/colour_v9.py train | apply <site_dir> [...]      env OUT_TAG, SMOOTH=9, THR=0.5, LAYER1=s2on_r2_oct_rn5"""
import json, os, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, joblib, lightgbm as lgb
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent)); os.environ.setdefault("COLOUR_TAG", "v8")
import colour_multi as C, qgis_style as Q, safe_resources  # noqa: E402
import site_paths as SP
P = C.P; F = list(C.COLS) + ["B4", "B5/B4", "B4/B3"]; ACRE = 0.0247105
MODEL = P / "sites/layer2/colour_model_v9.joblib"; THR = float(os.environ.get("THR", 0.5)); SMOOTH = int(os.environ.get("SMOOTH", 9))
L1 = os.environ.get("LAYER1", "s2on_r2_oct_rn5")


def feats11(b2, b3, b4, b5):
    X = pd.DataFrame(C.feats(b2, b3, b5), columns=C.COLS); X["B4"] = b4; X["B5/B4"] = b5 / (b4 + 1e-6); X["B4/B3"] = b4 / (b3 + 1e-6); return X


def sure_cane_stats(S, src, i, rng=np.random.default_rng(0)):
    """median / IQR of the 11 features over the image's sure-cane clear pixels (None if < 500 px)."""
    with rasterio.open(S / "model" / L1 / "cane_prob_2026.tif") as r: sure = r.read(1) >= 90
    l2 = S / "model/layer2_final/class3_2026.tif"
    if l2.exists():
        with rasterio.open(l2) as r: sure &= r.read(1) == 1
    cs = src["cs_cdf"].read(i + 1); ok = sure & (cs >= 60) & (cs <= 100); idx = np.flatnonzero(ok.ravel())
    if len(idx) < 500: return None
    idx = rng.choice(idx, min(len(idx), 200000), replace=False); b = {k: src[k].read(i + 1).ravel()[idx].astype("float32") / 1e4 for k in ("B2", "B3", "B4", "B5")}
    q = feats11(b["B2"], b["B3"], b["B4"], b["B5"]).quantile([.25, .5, .75]); return q.loc[.5], q.loc[.75] - q.loc[.25]


def train():
    S = pd.read_parquet(P / "sites/layer2/colour_separability_v2_b4/samples.parquet"); st = pickle.load(open(P / "sites/layer2/colour_image_stats_surecane.pkl", "rb"))
    S = S[[(s, d) in st for s, d in zip(S.site, S.date)]].reset_index(drop=True)
    for f in F: S[f] = [(v - st[(s, d)]["med"][f]) / (st[(s, d)]["iqr"][f] + 1e-6) for v, s, d in zip(S[f], S.site, S.date)]
    m = lgb.LGBMClassifier(**dict(C.PARAMS, n_jobs=safe_resources.cpus())).fit(S[F], S.y)
    joblib.dump({"model": m, "cols": F, "norm": "sure-cane median/IQR per image", "n": len(S)}, MODEL); print("saved", MODEL, len(S), "px")


def apply(site_dir):
    J = joblib.load(MODEL); m = J["model"]; m.set_params(n_jobs=safe_resources.cpus())
    S = SP.site_dir(site_dir) / "data"; st = S / "stacks"; o = S / "model" / os.environ.get("OUT_TAG", "colour_v9"); o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(S / "model" / L1 / "cane_map_2026.tif") as r: mask, prof = r.read(1), r.profile.copy()
    with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions)
    src = {b: rasterio.open(st / f"S2_{b}.tif") for b in ("B2", "B3", "B4", "B5", "cs_cdf")}
    p = np.full(mask.shape, np.nan, "float32"); used = []
    for dt in C.best_dates(site_dir):
        i = dates.index(dt); stats = sure_cane_stats(S, src, i)
        if stats is None: print("  skip", dt, "(too few sure-cane px)"); continue
        med, iqr = stats; cs = src["cs_cdf"].read(i + 1); b3 = src["B3"].read(i + 1)
        todo = (mask == 1) & (cs >= 60) & (cs <= 100) & (b3 > 0) & np.isnan(p)
        if not todo.any(): continue
        b = {k: src[k].read(i + 1)[todo].astype("float32") / 1e4 for k in ("B2", "B4", "B5")}; b["B3"] = b3[todo].astype("float32") / 1e4
        X = feats11(b["B2"], b["B3"], b["B4"], b["B5"])
        for f in F: X[f] = (X[f] - med[f]) / (iqr[f] + 1e-6)
        p[todo] = m.predict_proba(X[F])[:, 1]; used.append(dt)
    for s in src.values(): s.close()
    valid = (mask == 1) & np.isfinite(p)
    num = ndimage.uniform_filter(np.where(valid, p, 0), SMOOTH); den = ndimage.uniform_filter(valid.astype("float32"), SMOOTH)
    ps = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    cls = np.where(mask == 255, 65535, np.where(valid, np.where(ps >= THR, 1, 304), np.where(mask == 1, 65535, 0))).astype("uint16")
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "cane_colour_prob_2026.tif", "w", **pr) as d: d.write(np.where(valid, np.round(ps * 100), 255).astype("uint8"), 1)
    with rasterio.open(o / "class_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(cls, 1)
    Q.pseudocolor(o / "cane_colour_prob_2026.tif", Q.PROB_STOPS); Q.paletted(o / "class_2026.tif", {0: ("#e6e6e6", "none"), 1: ("#33ef2a", "sugarcane"), 304: ("#8b1a1a", "non-cane (cassava / rice / other)")})
    res = {"site": site_dir, "model": "colour_v9 (cane vs non-cane, 11 feats, sure-cane normalisation)", "images": used, "thr": THR, "smooth": SMOOTH, "layer1": L1,
           "cane_like_ac": round(float((mask == 1).sum() * ACRE)), "sugarcane_ac": round(float((cls == 1).sum() * ACRE)), "noncane_ac": round(float((cls == 304).sum() * ACRE)), "unknown_cloud_ac": round(float(((mask == 1) & ~valid).sum() * ACRE))}
    (o / "colour.json").write_text(json.dumps(res, indent=1)); print(res, flush=True); return res


if __name__ == "__main__":
    if sys.argv[1] == "train": train()
    else:
        for s in sys.argv[2:]: apply(s)
    print("COLOUR_V9_DONE")

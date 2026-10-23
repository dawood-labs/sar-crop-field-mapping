"""Rice stage 2/3 (user plan 16 Oct): the 5-3-2 COLOUR TRAJECTORY over Sep-Oct. Rice changes colour (light green -> dark green ->
yellow -> dull yellow -> harvest) within 1-2 months; cane stays in a shade of green.

Per pixel, from every clear date >= 10 Sep 2026 (stacks B2 B3 B4 B5 B8 + cs_cdf): yellowness y = (B4+B5)/2 / B3 (red+red-edge
over green: rises as a canopy turns yellow), greenness g = B8/B4 (NDVI-like), and the 5-3-2 hue. Trajectory features: y at
first / last clear date, y max, y slope (per 30 days), y range, g min, g drop (first - last), n clear dates.
Train: positives = stage-1 rice px (SAR dip), negatives = user cane polygons + reviewer sugarcane blocks (no dip); grouped CV.
Apply: to every sugarcane-class px WITHOUT a dip -> stage-3 rice (direct-seeded) where p >= THR.
Output <site>/data/model/rice_stage2/: rice_colour_prob_2026.tif, rice_all_2026.tif (1 = SAR dip, 2 = colour, 0 other, 255 out),
class4_2026.tif (1 sugarcane, 7 rice, 281 cassava), stage2.json. Model sites/layer2/rice_colour_model.joblib.
usage: python scripts/rice_stage2.py train [site..]   |   python scripts/rice_stage2.py apply [site..]
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd, joblib, lightgbm as lgb
from rasterio.features import rasterize
from scipy import ndimage
import site_paths as SP
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q  # noqa: E402
P = Path(__file__).resolve().parents[1]; START = "20260910"; THR = float(os.environ.get("RICE_THR", "0.7")); ACRE = 0.0247105
COLS = ["y_first", "y_last", "y_max", "y_slope", "y_range", "g_min", "g_drop", "g_last", "hue_last", "hue_drop", "n_clear"]
PARAMS = dict(n_estimators=300, num_leaves=15, min_child_samples=40, class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def trajectory(site, mask):
    """COLS features (dict of 2-D arrays, NaN where < 2 clear dates) for pixels in mask."""
    st = SP.site_dir(site) / "data/stacks"
    with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions)
    idx = [i for i, d in enumerate(dates) if d >= START]; t0 = pd.Timestamp(dates[idx[0]])
    H, W = mask.shape; rr, cc = np.nonzero(mask); N = len(rr)
    Y = np.full((len(idx), N), np.nan, "float32"); G = Y.copy(); HU = Y.copy(); T = np.array([(pd.Timestamp(dates[i]) - t0).days for i in idx], "float32")
    src = {b: rasterio.open(st / f"S2_{b}.tif") for b in ("B2", "B3", "B4", "B5", "B8", "cs_cdf")}
    for k, i in enumerate(idx):
        cs = src["cs_cdf"].read(i + 1)[rr, cc]; ok = (cs >= 60) & (cs <= 100)
        if not ok.any(): continue
        b = {n: src[n].read(i + 1)[rr, cc].astype("float32") / 10000 for n in ("B2", "B3", "B4", "B5", "B8")}; ok &= b["B8"] > 0
        Y[k, ok] = ((b["B4"] + b["B5"]) / 2 / (b["B3"] + 1e-6))[ok]; G[k, ok] = (b["B8"] / (b["B4"] + 1e-6))[ok]
        R, Gc, Bb = [np.clip((b[x] - 0.02) / 0.33, 0, 1) for x in ("B5", "B3", "B2")]; mx = np.maximum(np.maximum(R, Gc), Bb); mn = np.minimum(np.minimum(R, Gc), Bb); d = mx - mn + 1e-9
        hue = np.where(mx == R, ((Gc - Bb) / d) % 6, np.where(mx == Gc, (Bb - R) / d + 2, (R - Gc) / d + 4)) * 60; HU[k, ok] = hue[ok]
    for s in src.values(): s.close()
    n_clear = (~np.isnan(Y)).sum(0); f = {}
    def first_last(A):
        fi = np.argmax(~np.isnan(A), 0); la = A.shape[0] - 1 - np.argmax(~np.isnan(A[::-1]), 0)
        return np.take_along_axis(A, fi[None], 0)[0], np.take_along_axis(A, la[None], 0)[0], T[fi], T[la]
    yf, yl, tf, tl = first_last(Y); gf, gl, _, _ = first_last(G); hf, hl, _, _ = first_last(HU)
    f["y_first"], f["y_last"], f["y_max"] = yf, yl, np.nanmax(Y, 0); f["y_slope"] = (yl - yf) / np.maximum(tl - tf, 1) * 30; f["y_range"] = np.nanmax(Y, 0) - np.nanmin(Y, 0)
    f["g_min"], f["g_drop"], f["g_last"] = np.nanmin(G, 0), gf - gl, gl; f["hue_last"], f["hue_drop"] = hl, hf - hl; f["n_clear"] = n_clear.astype("float32")
    out = {}
    for k, v in f.items():
        a = np.full((H, W), np.nan, "float32"); v = np.where(n_clear >= 2, v, np.nan); a[rr, cc] = v; out[k] = a
    return out


def labels(site):
    """(mask, y) label pixels: 1 = stage-1 rice (SAR dip), 0 = user cane polygons + reviewer sugarcane blocks without dip."""
    S = SP.site_dir(site) / "data"
    with rasterio.open(S / "model/rice_stage1/rice_sar_2026.tif") as r: dip = r.read(1); tr = r.transform; H, W = r.height, r.width
    pos = dip == 1; neg = np.zeros((H, W), bool)
    key = {"site25_2026": "site25", "site41_2026": "Site41", "site33_2026": "Site33"}.get(site)
    polys = []
    for f, col, val in (("cane_polygons_2026-10-15.gpkg", "site", key), ("cane_polygons_site25_2026-10-15.gpkg", None, "site25_2026"), ("site27_gcps_2026-10-16.gpkg", "attribute", "cane")):
        g = gpd.read_file(P / "sites/user_labels" / f).to_crs(32648)
        if col is None: g = g if site == val else g.iloc[0:0]
        elif col == "attribute": g = g[g.attribute == "cane"] if site == "bench_site27_2026" else g.iloc[0:0]
        else: g = g[g.site == val]
        polys += list(g.geometry)
    if polys: neg |= rasterize([(p, 1) for p in polys], out_shape=(H, W), transform=tr, dtype="uint8").astype(bool)
    lab = pd.read_csv(P / "sites/layer2_labels_all.csv"); lab = lab[(lab.site == site) & (lab.what == "sugarcane") & (lab.answer == "cane") & lab.confidence.isin(["medium", "high"])]
    for _, q in lab.iterrows(): neg[q.row - 1:q.row + 2, q.col - 1:q.col + 2] = True
    neg &= (dip == 0)
    return pos, neg


def train(sites):
    rows = []
    for s in sites:
        pos, neg = labels(s)
        if pos.sum() == 0 or neg.sum() == 0: print(s, "skip (no labels)"); continue
        rng = np.random.default_rng(0); pr, pc = np.nonzero(pos); k = rng.choice(len(pr), min(3000, len(pr)), replace=False); pos2 = np.zeros_like(pos); pos2[pr[k], pc[k]] = True
        m = pos2 | neg; f = trajectory(s, m)
        df = pd.DataFrame({c: f[c][m] for c in COLS}); df["y"] = pos2[m].astype(int); df["site"] = s; df = df.dropna(subset=["y_first"]); rows.append(df)
        print(s, "px: rice", int(df.y.sum()), "cane", int((df.y == 0).sum()), flush=True)
    df = pd.concat(rows, ignore_index=True); df.to_parquet(P / "sites/layer2/rice_colour_train.parquet", index=False)
    print("\nLEAVE-ONE-DISTRICT-OUT (px): rice recall / cane false-rice at THR", THR)
    for s in df.site.unique():
        te = df[df.site == s]; tr_ = df[df.site != s]
        if te.y.nunique() < 2 or tr_.y.nunique() < 2: continue
        p = lgb.LGBMClassifier(**PARAMS).fit(tr_[COLS], tr_.y).predict_proba(te[COLS])[:, 1]
        print(f"  {s.replace('_2026', ''):18s} rice recall {(p[te.y == 1] >= THR).mean():.2f}   cane called rice {(p[te.y == 0] >= THR).mean():.2f}")
    m = lgb.LGBMClassifier(**PARAMS).fit(df[COLS], df.y); joblib.dump({"model": m, "cols": COLS}, P / "sites/layer2/rice_colour_model.joblib")
    print("importance:", dict(sorted(zip(COLS, m.feature_importances_.tolist()), key=lambda x: -x[1])[:6]))


def apply(sites):
    m = joblib.load(P / "sites/layer2/rice_colour_model.joblib")["model"]
    for s in sites:
        S = SP.site_dir(s) / "data"; o = S / "model/rice_stage2"; o.mkdir(parents=True, exist_ok=True)
        with rasterio.open(S / "model/layer2_final/class3_2026.tif") as r: cls, prof = r.read(1), r.profile.copy()
        with rasterio.open(S / "model/rice_stage1/rice_sar_2026.tif") as r: dip = r.read(1)
        cand = (cls == 1) & (dip == 0); f = trajectory(s, cand)
        X = pd.DataFrame({c: f[c][cand] for c in COLS}); ok = ~X.y_first.isna().values
        p = np.full(cls.shape, 255, "uint8"); pv = np.full(len(X), 255, "uint8"); pv[ok] = np.round(100 * m.predict_proba(X[ok])[:, 1]).astype("uint8"); p[cand] = pv
        ps = ndimage.median_filter(np.where(p == 255, 0, p).astype("float32"), 5)
        rice_c = cand & (p != 255) & (ps >= 100 * THR)
        allr = np.where(cls == 65535, 255, np.where(dip == 1, 1, np.where(rice_c, 2, 0))).astype("uint8")
        cls4 = np.where(cls == 65535, 65535, np.where((dip == 1) | rice_c, 7, cls)).astype("uint16")
        pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
        with rasterio.open(o / "rice_colour_prob_2026.tif", "w", **pr) as d: d.write(p, 1)
        with rasterio.open(o / "rice_all_2026.tif", "w", **pr) as d: d.write(allr, 1)
        with rasterio.open(o / "class4_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(cls4, 1)
        Q.pseudocolor(o / "rice_colour_prob_2026.tif", Q.PROB_STOPS)
        Q.paletted(o / "rice_all_2026.tif", {0: ("#e6e6e6", "sugarcane (no rice signal)"), 1: ("#1B5E20", "rice: SAR sowing dip"), 2: ("#66bb6a", "rice: colour trajectory (DSR)")})
        Q.paletted(o / "class4_2026.tif", {0: ("#e6e6e6", "none"), 1: ("#33ef2a", "sugarcane"), 7: ("#1B5E20", "rice"), 281: ("#8b1a1a", "cassava")})
        res = {"site": s, "sugarcane_before_ac": round(float((cls == 1).sum() * ACRE)), "rice_sar_ac": round(float((dip == 1).sum() * ACRE)), "rice_colour_ac": round(float(rice_c.sum() * ACRE)),
               "no_colour_data_ac": round(float((cand & (p == 255)).sum() * ACRE)), "sugarcane_after_ac": round(float((cls4 == 1).sum() * ACRE)), "thr": THR}
        (o / "stage2.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)
    print("STAGE2_DONE")


if __name__ == "__main__":
    sites = sys.argv[2:] or sorted(p.name for p in SP.all_sites("*_2026") if (p / "data/model/rice_stage1").exists())
    train(sites) if sys.argv[1] == "train" else apply(sites)

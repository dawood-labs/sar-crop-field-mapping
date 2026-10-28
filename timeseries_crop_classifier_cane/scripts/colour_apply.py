"""Global colour model (layer 2 cassava): trained on the user's cane / cassava polygons (site86 26+34, Site113 6+8, Site131
2+9) in the 5-3-2 bands of each district's late-season image; applied inside the cane-like mask of every district on the
district's clearest late-season image(s) (per pixel the latest clear date among the two best).
Output <site>/data/model/cassava_colour_v7/: cassava_prob_2026.tif, class_2026.tif (1 sugarcane, 281 cassava, 255 unknown/cloud),
class3_2026.tif (0 none, 1 sugarcane, 7 rice (from rn5_l2b), 281 cassava), colour.json. Model: sites/layer2/colour_model_v7.joblib.
usage: python scripts/colour_apply.py [site_dir ...]   (default: all 19)
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd, joblib, lightgbm as lgb
from rasterio.features import rasterize
from scipy import ndimage
import site_paths as SP
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q  # noqa: E402
P = Path(__file__).resolve().parents[1]; COLS = ["B5", "B3", "B2", "B5/B3", "B3/B2", "hue", "sat", "val"]; ACRE = 0.0247105
BEST = {"site136": ["20261012", "20261002"], "site82": ["20261012", "20261002"], "site06": ["20261007", "20261012"], "site84": ["20261007", "20261012"],
        "bench_site102": ["20261012", "20261007"], "site20": ["20261012", "20261007"], "site119": ["20261002", "20261012"], "site134": ["20261002", "20261012"],
        "bench_site26": ["20261004", "20260929"], "site112": ["20261002", "20261012"], "site122": ["20261002", "20261012"], "site86": ["20261012", "20261007"],
        "site129": ["20261007", "20261002"], "site139": ["20260930", "20261002"], "site12": ["20261002"], "site05": ["20261009", "20261004"], "site131": ["20261004", "20260919"],
        "site113": ["20261009", "20261006"], "site142": ["20260930", "20260920"]}
LABEL_IMG = {"site86": "20261012", "Site131": "20261004", "Site113": "20261009"}; SITE_DIR = {"site86": "site86_2026", "Site131": "site131_2026", "Site113": "site113_2026"}


def feats(b2, b3, b5):
    R, G, Bb = [np.clip((x - 0.02) / 0.33, 0, 1) for x in (b5, b3, b2)]
    mx = np.maximum(np.maximum(R, G), Bb); mn = np.minimum(np.minimum(R, G), Bb); d = mx - mn + 1e-9
    h = np.where(mx == R, ((G - Bb) / d) % 6, np.where(mx == G, (Bb - R) / d + 2, (R - G) / d + 4)) * 60
    return np.stack([b5, b3, b2, b5 / (b3 + 1e-6), b3 / (b2 + 1e-6), h, d / (mx + 1e-9), mx], -1)


def read_date(site_dir, dt):
    """(B2, B3, B5 reflectance, clear mask) for one date, from the raw per-date file or the stacks."""
    S = SP.site_dir(site_dir) / "data"; st = S / "stacks"
    with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions); cs = c.read(dates.index(dt) + 1)
    f = S / "raw/s2" / f"S2_{dt}.tif"
    if f.exists():
        with rasterio.open(f) as r: b2, b3, b5 = (r.read(k).astype("float32") / 10000 for k in (1, 2, 4))
    else:
        i = dates.index(dt) + 1; b2, b3, b5 = (rasterio.open(st / f"S2_{b}.tif").read(i).astype("float32") / 10000 for b in ("B2", "B3", "B5"))
    return b2, b3, b5, (cs >= 60) & (cs <= 100) & (b3 > 0)


def train():
    rows = []
    files = [("site86", P / "sites/user_labels/cane_polygons_site86_2026-10-15.gpkg", 0), ("site86", P / "sites/user_labels/cassava_noise_site86_2026-10-15.gpkg", 1)]
    g = gpd.read_file(P / "sites/user_labels/gcps_country_2026-10-15_analysis.gpkg"); g = g[~g.call2.str.startswith(("rice", "unclear"))]
    c = gpd.read_file(P / "sites/user_labels/cane_polygons_2026-10-15.gpkg")
    parts = [gpd.read_file(f).to_crs(32648).assign(site=s, y=y)[["site", "y", "geometry"]] for s, f, y in files] + [g.to_crs(32648).assign(y=1)[["site", "y", "geometry"]], c.to_crs(32648).assign(y=0)[["site", "y", "geometry"]]]
    polys = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=32648)
    cache = {}
    for sk, sd in SITE_DIR.items():
        b2, b3, b5, ok = read_date(sd, LABEL_IMG[sk]); tr = rasterio.open(SP.site_dir(sd) / "data/grid/index.tif").transform
        for _, q in polys[polys.site == sk].iterrows():
            mk = rasterize([(q.geometry, 1)], out_shape=b3.shape, transform=tr, dtype="uint8").astype(bool) & ok
            if mk.sum(): X = feats(b2[mk], b3[mk], b5[mk]); rows.append(pd.DataFrame(X, columns=COLS).assign(y=q.y, site=sk))
    df = pd.concat(rows, ignore_index=True); print("training px:", df.groupby(["site", "y"]).size().to_dict(), flush=True)
    m = lgb.LGBMClassifier(n_estimators=200, num_leaves=7, min_child_samples=20, class_weight="balanced", verbose=-1).fit(df[COLS], df.y)
    joblib.dump({"model": m, "cols": COLS}, P / "sites/layer2/colour_model_v7.joblib"); return m


def apply(m, site_dir):
    name = site_dir.replace("_2026", ""); S = SP.site_dir(site_dir) / "data"; o = S / "model/cassava_colour_v7"; o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(S / "model/s2on_r2_oct_rn5_l2b/cane_map_2026.tif") as r: mask, prof = r.read(1), r.profile.copy()
    with rasterio.open(S / "model/s2on_r2_oct_rn5_l2b/class_2026.tif") as r: l2 = r.read(1)
    p = np.full(mask.shape, 255, "uint8"); src = np.zeros(mask.shape, "uint8"); used = []
    for k, dt in enumerate(BEST[name], 1):                       # best image first, second image only where the first is cloudy
        try: b2, b3, b5, ok = read_date(site_dir, dt)
        except Exception as e: print("  skip", dt, e); continue
        todo = (mask == 1) & ok & (p == 255)
        if todo.any():
            X = feats(b2[todo], b3[todo], b5[todo]); p[todo] = np.round(100 * m.predict_proba(pd.DataFrame(X, columns=COLS))[:, 1]).astype("uint8"); src[todo] = k
        used.append(dt)
    ps = ndimage.median_filter(np.where(p == 255, 50, p).astype("float32"), 5)
    cls = np.where(mask == 255, 255, np.where((mask == 1) & (p != 255), np.where(ps >= 50, 281, 1), np.where(mask == 1, 255, 0))).astype("uint16")
    cls3 = np.where(l2 == 255, 65535, np.where(l2 == 7, 7, np.where(cls == 281, 281, np.where(mask == 1, 1, 0)))).astype("uint16")
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "cassava_prob_2026.tif", "w", **pr) as d: d.write(p, 1)
    with rasterio.open(o / "class_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(np.where(cls == 255, 65535, cls).astype("uint16"), 1)
    with rasterio.open(o / "class3_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(cls3, 1)
    Q.pseudocolor(o / "cassava_prob_2026.tif", Q.PROB_STOPS)
    Q.paletted(o / "class_2026.tif", {0: ("#e6e6e6", "non-cane"), 1: ("#33ef2a", "sugarcane"), 281: ("#8b1a1a", "cassava")})
    Q.paletted(o / "class3_2026.tif", {0: ("#e6e6e6", "none"), 1: ("#33ef2a", "sugarcane"), 7: ("#1B5E20", "rice"), 281: ("#8b1a1a", "cassava")})
    res = {"site": site_dir, "images": used, "cane_like_ac": round(float((mask == 1).sum() * ACRE)), "sugarcane_ac": round(float((cls == 1).sum() * ACRE)),
           "cassava_ac": round(float((cls == 281).sum() * ACRE)), "unknown_cloud_ac": round(float(((mask == 1) & (p == 255)).sum() * ACRE)), "rice_ac": round(float((l2 == 7).sum() * ACRE))}
    (o / "colour.json").write_text(json.dumps(res, indent=1)); print(res, flush=True); return res


if __name__ == "__main__":
    m = train() if not (P / "sites/layer2/colour_model_v7.joblib").exists() or os.environ.get("RETRAIN") == "1" else joblib.load(P / "sites/layer2/colour_model_v7.joblib")["model"]
    sites = sys.argv[1:] or [f"{n}_2026" for n in BEST]
    for s in sites: apply(m, s)
    print("COLOUR_DONE")

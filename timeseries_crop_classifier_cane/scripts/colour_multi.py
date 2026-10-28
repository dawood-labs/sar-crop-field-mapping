"""Colour model (layer 2: sugarcane = green vs cassava = maroon in 5-3-2) trained on the user's polygons sampled on EVERY
clear late-season date (>= 10 Sep 2026) of their districts, so the model sees the colour variation between images / tracks.
Then applied to all 19 districts (per pixel: latest clear date among the district's 3 clearest late-season images).
  python scripts/colour_multi.py train          -> sites/layer2/colour_model_v8.joblib + leave-one-date-out report
  python scripts/colour_multi.py apply [site..] -> <site>/data/model/cassava_colour_v8/ {cassava_prob, class (1/281), class3, colour.json}
  python scripts/colour_multi.py check          -> agreement with the user's 40 verified squares in site66
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd, joblib, lightgbm as lgb
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage
import site_paths as SP
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q  # noqa: E402
P = Path(__file__).resolve().parents[1]; COLS = ["B5", "B3", "B2", "B5/B3", "B3/B2", "hue", "sat", "val"]; ACRE = 0.0247105; START = "20260910"
SITE_DIR = {"site66": "site66_2026", "Site111": "site111_2026", "Site93": "site93_2026", "site109": "site109_2026", "site82": "bench_site82_2026"}
TAG = os.environ.get("COLOUR_TAG", "v8")            # v8 = cane vs cassava (layer2_final); v9 = cane vs NON-cane (cassava + rice + other; user 20 Oct 2026)
NONCANE = 281 if TAG == "v8" else 304               # output code of the non-cane class (281 cassava for v8, 304 "Other Crops" for v9)
PARAMS = dict(n_estimators=300, num_leaves=15, min_child_samples=40, class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)


def feats(b2, b3, b5):
    R, G, Bb = [np.clip((x - 0.02) / 0.33, 0, 1) for x in (b5, b3, b2)]
    mx = np.maximum(np.maximum(R, G), Bb); mn = np.minimum(np.minimum(R, G), Bb); d = mx - mn + 1e-9
    h = np.where(mx == R, ((G - Bb) / d) % 6, np.where(mx == G, (Bb - R) / d + 2, (R - G) / d + 4)) * 60
    return np.stack([b5, b3, b2, b5 / (b3 + 1e-6), b3 / (b2 + 1e-6), h, d / (mx + 1e-9), mx], -1)


def polygons():
    g = gpd.read_file(P / "sites/user_labels/gcps_country_2026-10-15_analysis.gpkg"); g = g[~g.call2.str.startswith(("rice", "unclear"))].to_crs(32648).assign(y=1)
    parts = [g[["site", "y", "geometry"]],
             gpd.read_file(P / "sites/user_labels/cassava_noise_site66_2026-10-15.gpkg").to_crs(32648).assign(site="site66", y=1)[["site", "y", "geometry"]],
             gpd.read_file(P / "sites/user_labels/cane_polygons_2026-10-15.gpkg").to_crs(32648).assign(y=0)[["site", "y", "geometry"]],
             gpd.read_file(P / "sites/user_labels/cane_polygons_site66_2026-10-15.gpkg").to_crs(32648).assign(site="site66", y=0)[["site", "y", "geometry"]]]
    if TAG != "v8":                                    # v9: non-cane = cassava + rice (+ anything the user marked not-cane)
        rd = lambda f: gpd.read_file(P / "sites/user_labels" / f).to_crs(32648)
        g2 = gpd.read_file(P / "sites/user_labels/gcps_country_2026-10-15_analysis.gpkg"); g2 = g2[g2.call2.str.startswith("rice")].to_crs(32648).assign(y=1)
        nb = rd("site82_gcps_2026-10-16.gpkg")
        parts += [g2[["site", "y", "geometry"]]] + ([] if os.environ.get("NO_SITE_RICE") else [rd("site109_rice_gcps_2026-10-16.gpkg").assign(site="site109", y=1)[["site", "y", "geometry"]]])
        parts += [rd("site109_cane_gcps_2026-10-16.gpkg").assign(site="site109", y=0)[["site", "y", "geometry"]],
                  nb.assign(site="site82", y=(nb.attribute == "not-cane").astype(int))[["site", "y", "geometry"]]]
    pl = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=32648); pl["pid"] = range(len(pl)); return pl


def train():
    pl = polygons(); rows = []
    for sk, sd in SITE_DIR.items():
        st = SP.site_dir(sd) / "data/stacks"
        with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions); tr = c.transform
        late = [i for i, d in enumerate(dates) if d >= START]
        src = {b: rasterio.open(st / f"S2_{b}.tif") for b in ("B2", "B3", "B5", "cs_cdf")}
        for _, q in pl[pl.site == sk].iterrows():
            x0, y0, x1, y1 = q.geometry.bounds; c0, r0 = ~tr * (x0, y1); c1, r1 = ~tr * (x1, y0)
            w = Window(int(c0), int(r0), int(np.ceil(c1)) - int(c0), int(np.ceil(r1)) - int(r0)); wt = rasterio.windows.transform(w, tr)
            mk = rasterize([(q.geometry, 1)], out_shape=(int(w.height), int(w.width)), transform=wt, dtype="uint8").astype(bool)
            cs = src["cs_cdf"].read([i + 1 for i in late], window=w); b2 = src["B2"].read([i + 1 for i in late], window=w); b3 = src["B3"].read([i + 1 for i in late], window=w); b5 = src["B5"].read([i + 1 for i in late], window=w)
            for k, i in enumerate(late):
                ok = mk & (cs[k] >= 60) & (cs[k] <= 100) & (b3[k] > 0)
                if ok.sum() < max(3, 0.6 * mk.sum()): continue
                X = feats(b2[k][ok] / 10000.0, b3[k][ok] / 10000.0, b5[k][ok] / 10000.0)
                rows.append(pd.DataFrame(X, columns=COLS).assign(y=q.y, pid=q.pid, site=sk, date=dates[i]))
        for s in src.values(): s.close()
    df = pd.concat(rows, ignore_index=True); df.to_parquet(P / "sites/layer2" / f"colour_{TAG}_train.parquet", index=False)
    print(f"training px by site/date (y=0 cane, y=1 {'cassava' if TAG == 'v8' else 'non-cane'}):"); print(df.groupby(["site", "date", "y"]).size().unstack(fill_value=0).to_string(), flush=True)
    # leave-one-date-out: train on all other dates (all sites), test the held-out date, polygon level
    print("\nLEAVE-ONE-DATE-OUT (polygon level):")
    for (sk, dt), te in df.groupby(["site", "date"]):
        tr_ = df[~((df.site == sk) & (df.date == dt))]; cap = int(os.environ.get("GROUP_CAP", "0"))
        w_ = np.minimum(1.0, cap / tr_.groupby(["site", "date", "y"]).y.transform("size")).values if cap else None
        m = lgb.LGBMClassifier(**PARAMS).fit(tr_[COLS], tr_.y, sample_weight=w_)
        pol = te.assign(p=m.predict_proba(te[COLS])[:, 1]).groupby("pid").agg(y=("y", "first"), p=("p", "mean"))
        print(f"  {sk:11s} {dt}: cane wrong {int(((pol.y == 0) & (pol.p >= 0.5)).sum())}/{int((pol.y == 0).sum())}  non-cane wrong {int(((pol.y == 1) & (pol.p < 0.5)).sum())}/{int(pol.y.sum())}", flush=True)
    cap = int(os.environ.get("GROUP_CAP", "0")); w = None
    if cap:                                            # v9c: no (site, date, class) group counts more than `cap` px (Site82 rice = 6,800 px per date)
        n = df.groupby(["site", "date", "y"]).y.transform("size"); w = np.minimum(1.0, cap / n).values; print("group cap", cap, "effective px:", round(float(w.sum())))
    m = lgb.LGBMClassifier(**PARAMS).fit(df[COLS], df.y, sample_weight=w); joblib.dump({"model": m, "cols": COLS}, P / "sites/layer2" / f"colour_model_{TAG}.joblib"); print("saved", TAG)


def best_dates(site_dir, n=3):
    S = SP.site_dir(site_dir) / "data"
    with rasterio.open(S / "stacks/S2_cs_cdf.tif") as c, rasterio.open(S / "grid/index.tif") as g:
        aoi = g.read(2) > 0; dates = list(c.descriptions); out = []
        for i, dt in enumerate(dates):
            if dt >= "20260915": cs = c.read(i + 1); out.append((dt, float(((cs >= 60) & (cs <= 100))[aoi].mean())))
    # prefer October dates with >= 70 % clear (September colours are less reliable), then the rest by clear share
    octo = [x for x in out if x[0] >= "20261001" and x[1] >= 0.7]; rest = [x for x in out if x not in octo]
    return [d for d, _ in sorted(octo, key=lambda x: -x[1]) + sorted(rest, key=lambda x: -x[1])][:n]


def apply(site_dir, tag=TAG):
    m = joblib.load(P / "sites/layer2" / f"colour_model_{tag}.joblib")["model"]; import safe_resources; m.set_params(n_jobs=safe_resources.cpus())   # cgroup quota aware (28 Oct)
    S = SP.site_dir(site_dir) / "data"; st = S / "stacks"; o = S / "model" / os.environ.get("OUT_TAG", f"cassava_colour_{tag}"); o.mkdir(parents=True, exist_ok=True)
    L1 = os.environ.get("LAYER1", "s2on_r2_oct_rn5")           # locked layer-1 output (user 16 Oct): rn5, rice removal set aside
    with rasterio.open(S / "model" / L1 / "cane_map_2026.tif") as r: mask, prof = r.read(1), r.profile.copy()
    l2 = np.where(mask == 255, 255, np.where(mask == 1, 1, 0)).astype("uint8")   # no rice class in the locked layer 1
    with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions)
    p = np.full(mask.shape, 255, "uint8"); used = []
    for dt in best_dates(site_dir):
        i = dates.index(dt) + 1; cs = rasterio.open(st / "S2_cs_cdf.tif").read(i); b3 = rasterio.open(st / "S2_B3.tif").read(i)
        todo = (mask == 1) & (cs >= 60) & (cs <= 100) & (b3 > 0) & (p == 255)
        if not todo.any(): continue
        b2 = rasterio.open(st / "S2_B2.tif").read(i); b5 = rasterio.open(st / "S2_B5.tif").read(i)
        X = feats(b2[todo] / 10000.0, b3[todo] / 10000.0, b5[todo] / 10000.0); p[todo] = np.round(100 * m.predict_proba(pd.DataFrame(X, columns=COLS))[:, 1]).astype("uint8"); used.append(dt)
    k = int(os.environ.get("SMOOTH", "5"))                 # 5 = median 5x5 (v8); 9 = 9x9 mean of p over the cane-like mask (final, mixed fields)
    if k == 5: ps = ndimage.median_filter(np.where(p == 255, 50, p).astype("float32"), 5)
    else:
        valid = (mask == 1) & (p != 255); num = ndimage.uniform_filter(np.where(valid, p, 0).astype("float32"), k); den = ndimage.uniform_filter(valid.astype("float32"), k)
        ps = np.where(den > 0, num / np.maximum(den, 1e-6), 50)
    cls = np.where(mask == 255, 65535, np.where((mask == 1) & (p != 255), np.where(ps >= 50, NONCANE, 1), np.where(mask == 1, 65535, 0))).astype("uint16")
    cls3 = np.where(l2 == 255, 65535, np.where(cls == NONCANE, NONCANE, np.where(mask == 1, 1, 0))).astype("uint16")
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "cassava_prob_2026.tif", "w", **pr) as d: d.write(p, 1)
    with rasterio.open(o / "class_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(cls, 1)
    with rasterio.open(o / "class3_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d: d.write(cls3, 1)
    Q.pseudocolor(o / "cassava_prob_2026.tif", Q.PROB_STOPS); NC = {281: ("#8b1a1a", "cassava"), 304: ("#8b1a1a", "non-cane (cassava / rice / other)")}[NONCANE]
    Q.paletted(o / "class_2026.tif", {0: ("#e6e6e6", "none"), 1: ("#33ef2a", "sugarcane"), NONCANE: NC}); Q.paletted(o / "class3_2026.tif", {0: ("#e6e6e6", "none"), 1: ("#33ef2a", "sugarcane"), NONCANE: NC})
    res = {"site": site_dir, "images": used, "cane_like_ac": round(float((mask == 1).sum() * ACRE)), "sugarcane_ac": round(float((cls == 1).sum() * ACRE)), "noncane_ac": round(float((cls == NONCANE).sum() * ACRE)), "noncane_code": NONCANE,
           "unknown_cloud_ac": round(float(((mask == 1) & (p == 255)).sum() * ACRE)), "layer1": L1}
    (o / "colour.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)


def check(tag=TAG):
    g = gpd.read_file(P / "sites/user_labels/checkset_v6c_site66_2026.gpkg"); g["user_call"] = g.user_call.fillna("").str.strip().str.lower().replace({"cane but bad health": "cane"})
    with rasterio.open(SP.site_dir("site66_2026") / "data/model" / f"cassava_colour_{tag}/class_2026.tif") as r: cls = r.read(1)
    g["model"] = [{1: "cane", 281: "cassava", 304: "cassava"}.get(int(cls[r.row, r.col]), "unknown") for _, r in g.iterrows()]
    print(pd.crosstab(g.user_call, g.model).to_string()); print(f"agreement {tag}: {(g.user_call == g.model).mean() * 100:.0f} %")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train()
    elif cmd == "apply":
        for s in (sys.argv[2:] or sorted(p.name for p in SP.all_sites("*_2026") if (p / "data/model/s2on_r2_oct_rn5_l2b").exists())): apply(s)
        print("COLOUR_DONE")
    else: check()

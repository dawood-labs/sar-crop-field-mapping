"""Static crop-type model (layer 2, cassava vs cane) with the user's polygons (15 Oct) as cassava labels.

Labels: cassava = user polygons (sites/user_labels/gcps_country_2026-10-15_analysis.gpkg, call2 cassava-like / "tall dense",
not cane per the user); cane = reviewed sugarcane 3x3 blocks (sites/layer2_labels_all.csv, confidence medium/high) + user
cane pixels (sites/user_labels.csv). Sites: site66, Site111, Site93 (Site111 / Site93 have no cane labels yet).
Features per pixel (full rasters, 5 x 5 texture):
  img_*   static S2 image (per site date, 5-3-2 image the user looked at): B2 B3 B4 B5 B8, NDVI, NDRE, GNDVI, 3x3 / 5x5 std
  opx_*   red-edge / SWIR latest clear from 15 Sep (data/raw/s2_extra): B6 B7 B8A B11 B12 + NDRE2, NDMI, NBR, SWIR ratio, 5x5 std
  sar_*   S1 Sep-Oct composite (per orbit VV VH VHmVV mean, temporal std, 5x5 texture) + Jun-Oct VH mean / max
  sea_*   season (pheno_2026.tif) bands
Evaluation: leave-field-out (GroupKFold 10) pixel and field level, plus leave-one-district-out (field level).
usage: python scripts/layer2_nsa.py            -> sites/layer2/nsa_points.parquet, nsa_cassava.csv
       APPLY=1 python scripts/layer2_nsa.py    -> + <site>/data/model/cassava_static_v0/{cassava_prob,class}_2026.tif (inspection only)
"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.features import rasterize
from scipy import ndimage
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import site_paths as SP
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q  # noqa: E402
P = Path(__file__).resolve().parents[1]; OUT = P / "sites/layer2"
SITES = {"site66": ("site66_2026", "20261012"), "Site111": ("site111_2026", "20261004"), "Site93": ("site93_2026", "20261009")}
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, class_weight="balanced", verbose=-1, random_state=0)


def tex(a, k):
    mu = ndimage.uniform_filter(a, k); mu2 = ndimage.uniform_filter(a * a, k); return np.sqrt(np.maximum(mu2 - mu * mu, 0))


def build_feats(S, img_date):
    st = S / "stacks"; feat = {}
    with rasterio.open(st / "S2_cs_cdf.tif") as r: dates = list(r.descriptions)
    di = dates.index(img_date) + 1
    rawf = S / "raw/s2" / f"S2_{img_date}.tif"
    if rawf.exists():
        with rasterio.open(rawf) as r: img = {b: r.read(i + 1).astype("float32") / 10000 for i, b in enumerate(("B2", "B3", "B4", "B5", "B8"))}
    else:
        img = {b: rasterio.open(st / f"S2_{b}.tif").read(di).astype("float32") / 10000 for b in ("B2", "B3", "B4", "B5", "B8")}
    cs_img = rasterio.open(st / "S2_cs_cdf.tif").read(di)
    nd = (img["B8"] - img["B4"]) / (img["B8"] + img["B4"] + 1e-6); ndre = (img["B8"] - img["B5"]) / (img["B8"] + img["B5"] + 1e-6); gnd = (img["B8"] - img["B3"]) / (img["B8"] + img["B3"] + 1e-6)
    for b in img: feat[f"img_{b}"] = img[b]
    feat["img_ndvi"], feat["img_ndre"], feat["img_gndvi"] = nd, ndre, gnd
    for k in (3, 5):
        feat[f"img_ndvi_tex{k}"] = tex(nd, k); feat[f"img_b8_tex{k}"] = tex(img["B8"], k); feat[f"img_b5_tex{k}"] = tex(img["B5"], k)
    feat["img_clear"] = ((cs_img >= 60) & (cs_img <= 100)).astype("float32")
    sx = sorted((S / "raw/s2_extra").glob("S2X_*.tif")) if (S / "raw/s2_extra").exists() else []
    sx = [f for f in sx if f.stem[4:] in dates]
    if sx:
        cube = np.stack([rasterio.open(f).read().astype("float32") for f in sx])
        csx = np.stack([rasterio.open(st / "S2_cs_cdf.tif").read(dates.index(f.stem[4:]) + 1) for f in sx])
        ok = (csx >= 60) & (csx <= 100) & (cube[:, 2] > 0)
        last = np.where(ok.any(0), ok.shape[0] - 1 - np.argmax(ok[::-1], 0), 0); none = ~ok.any(0)
        comp = {n: np.take_along_axis(cube[:, i], last[None], 0)[0] / 10000 for i, n in enumerate(("B6", "B7", "B8A", "B11", "B12"))}
        for n in comp: comp[n][none] = np.nan; feat[f"opx_{n}"] = comp[n]
        feat["opx_ndre2"] = (comp["B8A"] - comp["B6"]) / (comp["B8A"] + comp["B6"] + 1e-6); feat["opx_ndmi"] = (comp["B8A"] - comp["B11"]) / (comp["B8A"] + comp["B11"] + 1e-6)
        feat["opx_nbr"] = (comp["B8A"] - comp["B12"]) / (comp["B8A"] + comp["B12"] + 1e-6); feat["opx_swratio"] = comp["B11"] / (comp["B8A"] + 1e-6)
        feat["opx_ndmi_tex5"] = tex(np.nan_to_num(feat["opx_ndmi"]), 5); feat["opx_b11_tex5"] = tex(np.nan_to_num(comp["B11"]), 5)
        del cube, csx
    with rasterio.open(S / "features/s1_15d_2026nov.tif") as r: a1, n1 = r.read().astype("float32"), list(r.descriptions)
    a1[a1 <= -32000] = np.nan; a1 /= 100
    for orb, tag in (("RO164_DES", "des"), ("RO099_ASC", "asc")):
        for v in ("VV", "VH", "VHmVV"):
            so = np.stack([a1[n1.index(f"{orb}_{v}_{p}")] for p in ("202609a", "202609b", "202610a")])
            feat[f"sar_{tag}_{v}_mean"] = np.nanmean(so, 0); feat[f"sar_{tag}_{v}_tstd"] = np.nanstd(so, 0); feat[f"sar_{tag}_{v}_tex5"] = tex(np.nan_to_num(np.nanmean(so, 0)), 5)
        vhjo = np.stack([a1[n1.index(f"{orb}_VH_{p}")] for p in ("202606a", "202606b", "202607a", "202607b", "202608a", "202608b", "202609a", "202609b", "202610a")])
        feat[f"sar_{tag}_VH_junoct_mean"] = np.nanmean(vhjo, 0); feat[f"sar_{tag}_VH_junoct_max"] = np.nanmax(vhjo, 0)
    del a1
    with rasterio.open(S / "features/pheno_2026.tif") as r: ph, nph = r.read().astype("float32"), list(r.descriptions)
    ph[ph <= -32000] = np.nan
    for i, n in enumerate(nph):
        feat[f"sea_{n}"] = ph[i] if (n.split("_", 2)[-1] in ("last_cut", "grow_len", "grow_total") or n in ("greenup_days", "n_clear_mayoct")) else ph[i] / 100
    return feat


def label_pixels(site_key, site_dir, H, W, TR):
    pix = []
    g = gpd.read_file(P / "sites/user_labels/gcps_country_2026-10-15_analysis.gpkg").to_crs(32648)
    for _, q in g[g.site == site_key].iterrows():
        if q.call2.startswith("unclear") or q.call2.startswith("rice") or q.call2.startswith("bare"): continue
        mk = rasterize([(q.geometry, 1)], out_shape=(H, W), transform=TR, dtype="uint8").astype(bool); rr, cc = np.nonzero(mk)
        pix += [(r, c, 1, f"poly{q.id}") for r, c in zip(rr, cc)]
    extra = P / "sites/user_labels/cassava_noise_site66_2026-10-15.gpkg"          # 14 more cassava fields (user, 15 Oct 07:54)
    if site_key == "site66" and extra.exists():
        for j, q in gpd.read_file(extra).to_crs(32648).iterrows():
            mk = rasterize([(q.geometry, 1)], out_shape=(H, W), transform=TR, dtype="uint8").astype(bool); rr, cc = np.nonzero(mk)
            pix += [(r, c, 1, f"cas2_{j}") for r, c in zip(rr, cc)]
    canef = P / "sites/user_labels/cane_polygons_2026-10-15.gpkg"                      # user cane polygons (Site93 6, Site111 2)
    if canef.exists():
        gc = gpd.read_file(canef).to_crs(32648)
        if os.environ.get("CANE_SURE") == "1": gc = gc[gc.confidence != "uncertain"]
        for j, q in gc[gc.site == site_key].iterrows():
            mk = rasterize([(q.geometry, 1)], out_shape=(H, W), transform=TR, dtype="uint8").astype(bool); rr, cc = np.nonzero(mk)
            pix += [(r, c, 0, f"ucane{j}") for r, c in zip(rr, cc)]
    lab = pd.read_csv(P / "sites/layer2_labels_all.csv"); lab = lab[(lab.site == site_dir) & (lab.what == "sugarcane") & (lab.answer == "cane") & lab.confidence.isin(["medium", "high"])]
    if os.environ.get("USER_ONLY") == "1": lab = lab.iloc[0:0]          # mimic the user's eye: only the user's own labels
    for i, q in lab.iterrows(): pix += [(q.row + dr, q.col + dc, 0, f"blk{i}") for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    ul = pd.read_csv(P / "sites/user_labels.csv"); ul = ul[ul.site == site_dir]
    for i, q in ul.iterrows(): pix += [(q.row + dr, q.col + dc, 0, f"user{i}") for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    pix = pd.DataFrame(pix, columns=["row", "col", "y", "group"]).drop_duplicates(["row", "col"])
    pix["group"] = site_key + ":" + pix.group; pix["site"] = site_key
    return pix


def main():
    parts = []
    for site_key, (site_dir, img_date) in SITES.items():
        S = SP.site_dir(site_dir) / "data"
        with rasterio.open(S / "grid/index.tif") as ix: TR, H, W = ix.transform, ix.height, ix.width
        pix = label_pixels(site_key, site_dir, H, W, TR); feat = build_feats(S, img_date)
        rows = {k: v[pix.row.values, pix.col.values] for k, v in feat.items()}; del feat
        parts.append(pd.concat([pix.reset_index(drop=True), pd.DataFrame(rows)], axis=1))
        print(site_key, "label px:", pix.y.value_counts().to_dict(), "groups:", pix.group.nunique(), flush=True)
    df = pd.concat(parts, ignore_index=True); df.to_parquet(OUT / "nsa_points.parquet", index=False)
    groups = {"img (static image)": [c for c in df.columns if c.startswith("img_")], "opx (red-edge/SWIR)": [c for c in df.columns if c.startswith("opx_")],
              "sar (Sep-Oct composite)": [c for c in df.columns if c.startswith("sar_")], "sea (season)": [c for c in df.columns if c.startswith("sea_")]}
    groups["img+sar"] = groups["img (static image)"] + groups["sar (Sep-Oct composite)"]
    groups["all"] = groups["img (static image)"] + groups["opx (red-edge/SWIR)"] + groups["sar (Sep-Oct composite)"] + groups["sea (season)"]
    res = []
    for name, cols in groups.items():
        oof = np.zeros(len(df))
        for trn, val in GroupKFold(10).split(df, df.y, df.group):
            oof[val] = lgb.LGBMClassifier(**PARAMS).fit(df.iloc[trn][cols], df.y.iloc[trn]).predict_proba(df.iloc[val][cols])[:, 1]
        p, r, f, _ = precision_recall_fscore_support(df.y, oof >= 0.5, average="binary")
        pol = df.assign(p=oof).groupby("group").agg(y=("y", "first"), p=("p", "mean")); pp, pr_, pf, _ = precision_recall_fscore_support(pol.y, pol.p >= 0.5, average="binary")
        lodo = np.full(len(df), np.nan)
        for st_ in df.site.unique():
            m = (df.site == st_).values
            if df.y[~m].nunique() < 2: continue
            lodo[m] = lgb.LGBMClassifier(**PARAMS).fit(df[~m][cols], df.y[~m]).predict_proba(df[m][cols])[:, 1]
        pl = df.assign(p=lodo).groupby("group").agg(y=("y", "first"), p=("p", "mean"), site=("site", "first")).dropna()
        if pl.y.nunique() == 2: lp, lr, _, _ = precision_recall_fscore_support(pl.y, pl.p >= 0.5, average="binary")
        else: lp = lr = np.nan
        res.append(dict(features=name, auc=round(roc_auc_score(df.y, oof), 3), px_P=round(p, 2), px_R=round(r, 2), field_P=round(pp, 2), field_R=round(pr_, 2), LODO_field_P=None if lp != lp else round(lp, 2), LODO_field_R=None if lr != lr else round(lr, 2)))
        if name in ("all", "sea (season)"):
            print(f"LODO per district ({name}):"); print(pl.assign(yh=(pl.p >= 0.5).astype(int)).groupby(["site", "y"]).yh.agg(["size", "mean"]).round(2).to_string())
        if name == "all":
            mdl = lgb.LGBMClassifier(**PARAMS).fit(df[cols], df.y); imp = pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False); print("top features:", imp.head(10).to_dict())
    res = pd.DataFrame(res); res.to_csv(OUT / "nsa_cassava.csv", index=False); pd.set_option("display.width", 200); print(res.to_string(index=False))
    if os.environ.get("APPLY") == "1":
        cols = groups["all"]; mdl = lgb.LGBMClassifier(**PARAMS).fit(df[cols], df.y)
        for site_key, (site_dir, img_date) in SITES.items():
            S = SP.site_dir(site_dir) / "data"; o = S / ("model/cassava_static_" + os.environ.get("VTAG", "v2")); o.mkdir(parents=True, exist_ok=True)
            with rasterio.open(S / "model/s2on_r2_oct_rn5_l2b/cane_map_2026.tif") as r: m, prof = r.read(1), r.profile.copy()
            feat = build_feats(S, img_date); idx = np.flatnonzero(m.ravel() == 1)
            X = np.stack([feat[c].ravel()[idx] if c in feat else np.full(len(idx), np.nan, "float32") for c in cols], 1)
            lab_px = df[df.site == site_key]
            if len(lab_px):                                           # sanity: model on the label pixels through the raster path
                chk = np.stack([feat[c][lab_px.row.values, lab_px.col.values] if c in feat else np.full(len(lab_px), np.nan, "float32") for c in cols], 1)
                pc = mdl.predict_proba(chk)[:, 1]; print(f"   check {site_key}: label px mean p(cassava) y=1 {pc[lab_px.y.values == 1].mean():.2f}  y=0 {pc[lab_px.y.values == 0].mean():.2f}", flush=True)
            del feat
            p = np.full(m.size, 255, "uint8"); p[idx] = np.round(100 * mdl.predict_proba(X)[:, 1]).astype("uint8"); p = p.reshape(m.shape)
            ps = ndimage.median_filter(np.where(p == 255, 0, p).astype("float32"), 5)          # 5x5 median: within-field noise
            cls = np.where(m == 255, 65535, np.where((m == 1) & (p != 255) & (ps >= 50), 281, np.where(m == 1, 1, 0))).astype("uint16")
            pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
            with rasterio.open(o / "cassava_prob_2026.tif", "w", **pr) as d_: d_.write(p, 1)
            with rasterio.open(o / "class_2026.tif", "w", **dict(pr, dtype="uint16", nodata=65535)) as d_: d_.write(cls, 1)
            Q.pseudocolor(o / "cassava_prob_2026.tif", Q.PROB_STOPS); Q.paletted(o / "class_2026.tif", {0: ("#e6e6e6", "non-cane"), 1: ("#33ef2a", "sugarcane"), 281: ("#ff9800", "cassava candidate (static)")})
            ac = 0.0247105; print(f"{site_key}: cane-like {int((m == 1).sum() * ac):,} ac -> cassava candidate {int((cls == 281).sum() * ac):,} ac, sugarcane {int((cls == 1).sum() * ac):,} ac -> {o}", flush=True)


if __name__ == "__main__":
    main()

"""Analyse the user's QGIS polygons (sites/user_labels/gcps_country_2026-10-15.gpkg, column site): per polygon the
layer-1 / layer-2 map classes, phenology metrics and a rule-based crop call (rice / bare / cassava / cane-like / unclear).
Output: sites/user_labels/gcps_country_2026-10-15_analysis.gpkg + .csv  (one row per polygon, id = row order).
usage: python scripts/user_polygons.py
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import site_paths as SP
warnings.filterwarnings("ignore")
P = Path(__file__).resolve().parents[1]; F = P / "sites/user_labels/gcps_country_2026-10-15.gpkg"
SITE = {"site25": "site25_2026", "Site41": "site41_2026", "Site33": "site33_2026"}
g = gpd.read_file(F); g["id"] = range(len(g)); g = g.to_crs(32648)
rows = []
for site_key, site in SITE.items():
    sub = g[g.site == site_key]
    if not len(sub): continue
    d = SP.site_dir(site) / "data"
    with rasterio.open(d / "grid/index.tif") as ix: tr, H, W = ix.transform, ix.height, ix.width
    ph = rasterio.open(d / "features/pheno_2026.tif"); nph = list(ph.descriptions)
    m1 = rasterio.open(d / "model/s2on_r2_oct_rn5/cane_source_2026.tif"); rp = rasterio.open(d / "model/s2on_r2_oct_rn5_l2b/rice_prob_2026.tif")
    om = rasterio.open(d / "model/s2on_r2_oct/oct_ndvi_min.tif")
    s1 = rasterio.open(d / "features/s1_15d_2026nov.tif"); n1 = list(s1.descriptions)
    st = d / "stacks"; b4 = rasterio.open(st / "S2_B4.tif"); b8 = rasterio.open(st / "S2_B8.tif"); cs = rasterio.open(st / "S2_cs_cdf.tif")
    dates = pd.to_datetime(list(cs.descriptions))
    for _, r in sub.iterrows():
        x0, y0, x1, y1 = r.geometry.bounds
        c0, r0 = ~tr * (x0, y1); c1, r1 = ~tr * (x1, y0)
        c0, r0, c1, r1 = int(np.floor(c0)), int(np.floor(r0)), int(np.ceil(c1)), int(np.ceil(r1))
        w = Window(c0, r0, c1 - c0, r1 - r0); wt = rasterio.windows.transform(w, tr)
        mask = rasterize([(r.geometry, 1)], out_shape=(r1 - r0, c1 - c0), transform=wt, dtype="uint8").astype(bool)
        if mask.sum() == 0: continue
        o = {"id": int(r.id), "site": site_key, "n_px": int(mask.sum()), "area_ac": round(mask.sum() * 0.0247105, 2)}
        src = m1.read(1, window=w)[mask]; o["l1_model_%"] = round(100 * (src == 1).mean()); o["l1_rule_%"] = round(100 * (src == 2).mean()); o["l1_none_%"] = round(100 * (src == 0).mean())
        pr = rp.read(1, window=w)[mask].astype(float); pr[pr == 255] = np.nan; o["rice_prob"] = round(np.nanmean(pr)) if np.isfinite(pr).any() else None
        oc = om.read(1, window=w)[mask].astype(float); oc[oc == 255] = np.nan; o["oct_ndvi_min"] = round(np.nanmean(oc) / 100, 2) if np.isfinite(oc).any() else None
        v = ph.read(window=w).astype(float); v[v <= -32000] = np.nan; vm = np.nanmean(v[:, mask], axis=1)
        for k in ["ndvi_dry_min", "ndvi_marapr_max", "ndvi_mayjun_max", "ndvi_julsep_med", "ndvi_oct_med", "ndvi_max", "greenup_days", "RO164_DES_grow_len", "RO164_DES_end_vh", "RO164_DES_flood", "RO099_ASC_flood", "RO164_DES_dip", "RO164_DES_ystd", "RO164_DES_last_cut"]:
            x = vm[nph.index(k)]; o[k.replace("RO164_DES_", "D_").replace("RO099_ASC_", "A_").replace("ndvi_", "")] = round(x if (k.endswith(("grow_len", "greenup_days", "last_cut"))) else x / 100, 2)
        # NDVI series May - Oct (polygon mean, clear px >= 60 % of polygon)
        idx = [i for i, dt in enumerate(dates) if dt >= pd.Timestamp("2026-05-01")]
        ser = {}
        for i in idx:
            c_ = cs.read(i + 1, window=w)[mask]; ok = (c_ >= 60) & (c_ <= 100)
            if ok.mean() >= 0.6:
                a4 = b4.read(i + 1, window=w)[mask][ok].astype(float); a8 = b8.read(i + 1, window=w)[mask][ok].astype(float)
                ser[dates[i].strftime("%m-%d")] = round(float(np.mean((a8 - a4) / np.maximum(a8 + a4, 1))), 2)
        o["ndvi_series"] = " ".join(f"{k}:{v_}" for k, v_ in ser.items())
        vh = s1.read(window=w).astype(float); vh[vh <= -32000] = np.nan; vhm = np.nanmean(vh[:, mask], axis=1) / 100
        o["vh_jun_oct"] = " ".join(f"{p[-5:]}:{vhm[n1.index(f'RO164_DES_VH_{p}')]:.1f}" for p in ["202606a", "202606b", "202607a", "202607b", "202608a", "202608b", "202609a", "202609b", "202610a"])
        # rule-based call
        flood = min(o["D_flood"], o["A_flood"]); oct_ = o["oct_med"] if o["oct_med"] == o["oct_med"] else -1
        if (o["oct_ndvi_min"] is not None and o["oct_ndvi_min"] < 0.30) or (oct_ >= 0 and oct_ < 0.35): call = "bare / harvested in Oct"
        elif flood <= -20.5: call = "rice (flooded)"
        elif o["greenup_days"] >= 265 and oct_ >= 0.55: call = "rice-like (late green-up, no flood: DSR?)"
        elif o["julsep_med"] >= 0.55 and 120 <= o["greenup_days"] < 265 and o["D_end_vh"] >= -16.5: call = "cassava / cane-like (upland, early green-up)"
        else: call = "unclear"
        o["call"] = call; rows.append(o)
    for f_ in (ph, m1, rp, om, s1, b4, b8, cs): f_.close()
df = pd.DataFrame(rows); out = g.merge(df, on=["id", "site"]); out.to_file(F.with_name(F.stem + "_analysis.gpkg"), driver="GPKG")
df.to_csv(F.with_name(F.stem + "_analysis.csv"), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 60)
print(df[["id", "site", "area_ac", "l1_model_%", "l1_rule_%", "l1_none_%", "rice_prob", "oct_ndvi_min", "dry_min", "mayjun_max", "julsep_med", "oct_med", "greenup_days", "D_end_vh", "D_flood", "A_flood", "D_grow_len", "call"]].to_string(index=False))
print(); print(df.groupby(["site", "call"]).size().to_string())

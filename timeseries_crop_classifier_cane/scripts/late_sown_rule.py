"""Late-sown cane recall rule (v8, 21 Oct 2026; manager: the time-series model misses cane sown May-June, e.g. after a dry-season
second crop). Adds pixels the model (+ Oct rule + recall v5) left at 0 when the field was BARE in Apr-May, greened up between
15 May and 15 Aug, is dense Jul-Sep and denser in October, has a canopy-like SAR season end and NO flooding (user's rule:
relative VH dip, >= 2 consecutive 15-day composites, either track) - so transplanted rice stays out.
  aprmay_min   < 0.25     NDVI min of clear dates 1 Apr - 31 May
  greenup2     set        first clear date 15 May - 15 Aug with NDVI >= 0.5
  julsep_med  >= 0.55     NDVI median 1 Jul - 30 Sep
  oct_med     >= 0.70     NDVI median from 15 Sep
  end_vh      >= -16 dB   mean of both orbits (pheno); ystd >= 1.5 dB (crop-like year)
  oct_ndvi_min in [0.30, 1]  (October bare rule kept)
  flood        none       15-day VH anomaly (vs pixel median) <= -4 dB on >= 2 consecutive composites Jun-Sep, either track
Output <site>/data/model/<out tag>/: cane_map_2026.tif, cane_source_2026.tif (1 model, 2 rule v5, 3 late-sown rule), cane_prob /
oct_ndvi_min copies, late_rule.json (+ .qml). usage: python scripts/late_sown_rule.py <src tag> <out tag> sites/<site> [...]"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio
import site_paths as SP
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q  # noqa: E402
ACRE = 0.0247105


def ndvi_feats(S):
    with rasterio.open(S / "stacks/S2_cs_cdf.tif") as c: dates = list(c.descriptions); H, W = c.height, c.width
    aprmay = np.full((H, W), np.nan, "float32"); gu = np.full((H, W), np.nan, "float32"); js, om = [], []
    with rasterio.open(S / "stacks/S2_cs_cdf.tif") as c, rasterio.open(S / "stacks/S2_B4.tif") as r4, rasterio.open(S / "stacks/S2_B8.tif") as r8:
        for i, d in enumerate(dates):
            dd = pd.Timestamp(d)
            if dd < pd.Timestamp("2026-04-01"): continue
            cs = c.read(i + 1); b4 = r4.read(i + 1).astype("float32"); b8 = r8.read(i + 1).astype("float32")
            ok = (cs >= 60) & (cs <= 100) & (b8 > 0); n = (b8 - b4) / np.maximum(b8 + b4, 1); n[~ok] = np.nan
            if dd <= pd.Timestamp("2026-05-31"): aprmay = np.fmin(aprmay, n)
            if pd.Timestamp("2026-05-15") <= dd <= pd.Timestamp("2026-08-15"): g = np.isnan(gu) & (n >= 0.5); gu[g] = dd.dayofyear
            if pd.Timestamp("2026-07-01") <= dd <= pd.Timestamp("2026-09-30"): js.append(n)
            if dd >= pd.Timestamp("2026-09-15"): om.append(n)
    with np.errstate(all="ignore"): return aprmay, gu, np.nanmedian(np.stack(js), 0), np.nanmedian(np.stack(om), 0)


def flood_mask(S):
    with rasterio.open(S / "features/s1_15d_2026nov.tif") as r:
        nm = list(r.descriptions); flood = np.zeros((r.height, r.width), bool)
        for orb in ("RO164_DES", "RO099_ASC"):
            per = [n for n in nm if n.startswith(f"{orb}_VH_")]; vh = np.stack([r.read(nm.index(n) + 1) for n in per]).astype("float32"); vh[vh <= -32000] = np.nan; vh /= 100
            with np.errstate(all="ignore"): a = vh - np.nanmedian(vh, 0)
            run = np.zeros(flood.shape, int); best = np.zeros(flood.shape, int)
            for k, n in enumerate(per):
                if "202606" <= n[-7:-1] <= "202609": run = np.where(a[k] <= -4, run + 1, 0); best = np.maximum(best, run)
            flood |= best >= 2
    return flood


def one(site, src, dst):
    site = SP.site_dir(site); S = site / "data"; s = S / "model" / src; o = S / "model" / dst; o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(s / "cane_map_2026.tif") as r: m, prof = r.read(1), r.profile.copy()
    with rasterio.open(s / "cane_source_2026.tif") as r: src_b = r.read(1)
    with rasterio.open(s / "oct_ndvi_min.tif") as r: octmin = r.read(1)
    with rasterio.open(S / "features/pheno_2026.tif") as r:
        n = list(r.descriptions); g = lambda k: r.read(n.index(k) + 1).astype("float32")  # noqa: E731
        ev = (g("RO164_DES_end_vh") + g("RO099_ASC_end_vh")) / 2; ys = (g("RO164_DES_ystd") + g("RO099_ASC_ystd")) / 2
    aprmay, gu, js, om = ndvi_feats(S); flood = flood_mask(S)
    ok = np.isfinite(aprmay) & np.isfinite(js) & np.isfinite(om)
    rule = ok & (aprmay < 0.25) & np.isfinite(gu) & (js >= 0.55) & (om >= 0.70) & (ev >= -1600) & (ys >= 150) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    add = rule & (m == 0) & ~flood; excl = rule & (m == 0) & flood
    new = np.where(add, 1, m).astype("uint8"); src_n = np.where(add, 3, src_b).astype("uint8")
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "cane_map_2026.tif", "w", **pr) as d: d.write(new, 1)
    with rasterio.open(o / "cane_source_2026.tif", "w", **pr) as d: d.write(src_n, 1)
    with rasterio.open(o / "late_flood_excluded_2026.tif", "w", **pr) as d: d.write(np.where(m == 255, 255, excl.astype("uint8")), 1)
    for f in ("cane_prob_2026.tif", "cane_prob_2026.qml", "oct_ndvi_min.tif", "oct_ndvi_min.qml"):
        if (s / f).exists(): shutil.copy(s / f, o / f)
    Q.paletted(o / "cane_map_2026.tif", Q.MAP_CLASSES)
    Q.paletted(o / "cane_source_2026.tif", {0: ("#e6e6e6", "non-cane"), 1: ("#33ef2a", "cane: model"), 2: ("#ff8c00", "cane: recall rule v5"), 3: ("#8e24aa", "cane: late-sown rule v8 (added)")})
    Q.paletted(o / "late_flood_excluded_2026.tif", {0: ("#e6e6e6", "-"), 1: ("#1B5E20", "late-sown pattern but flooded (kept out: rice?)")})
    res = {"site": site.name, "src": src, "cane_ac_before": round(float((m == 1).sum() * ACRE)), "added_late_sown_ac": round(float(add.sum() * ACRE)),
           "excluded_flooded_ac": round(float(excl.sum() * ACRE)), "cane_ac_after": round(float((new == 1).sum() * ACRE)),
           "rule": "v8 late-sown: AprMay NDVI min<0.25 & greenup 15May-15Aug & JulSep med>=0.55 & Oct med>=0.70 & end_vh>=-16 & ystd>=1.5 & Oct not bare & no flood run (VH anomaly<=-4, 2 consecutive 15d, Jun-Sep)"}
    (o / "late_rule.json").write_text(json.dumps(res, indent=1)); print(res, flush=True); return res


if __name__ == "__main__":
    for p in sys.argv[3:]: one(p, sys.argv[1], sys.argv[2])

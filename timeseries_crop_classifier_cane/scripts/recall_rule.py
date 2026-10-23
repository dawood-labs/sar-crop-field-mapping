"""Recall net (round 3): add pixels with a clear S2 plant-cane pattern that the model map missed.
Rule (pheno_<year>.tif): ndvi_dry_min < 0.30 AND ndvi_mayjun_max >= 0.55 AND ndvi_julsep_med >= 0.60 AND greenup_days <= 250,
and not removed by the October bare rule (oct_ndvi_min >= 0.30). Test set site25 (14 Oct): R 0.47 -> 0.71, P 0.94 -> 0.96.
usage: python scripts/recall_rule.py <src tag (with oct rule)> <out tag> sites/<site> [...]
Output <site>/data/model/<out tag>/: cane_map_2026.tif (1 = cane), cane_source_2026.tif (1 model, 2 rule) + .qml, cane_prob_2026.tif copy, rule.json
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qgis_style as Q  # noqa: E402
import site_paths as SP
ACRE = 100 / 4046.8564224

def one(site, src, dst, year=2026):
    site = SP.site_dir(site); s = site / "data/model" / src; o = site / "data/model" / dst; o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(s / "cane_map_2026.tif") as r: m, prof = r.read(1), r.profile.copy()
    with rasterio.open(s / "oct_ndvi_min.tif") as r: octmin = r.read(1)
    with rasterio.open(site / "data/features" / f"pheno_{year}.tif") as r:
        n = list(r.descriptions); g = lambda k: r.read(n.index(k) + 1).astype("float32")  # noqa: E731
        dry, ma, mj, js, oc, gu = g("ndvi_dry_min"), g("ndvi_marapr_max"), g("ndvi_mayjun_max"), g("ndvi_julsep_med"), g("ndvi_oct_med"), g("greenup_days")
        ev = (g("RO164_DES_end_vh") + g("RO099_ASC_end_vh")) / 2
        ys = (g("RO164_DES_ystd") + g("RO099_ASC_ystd")) / 2
        fl = np.minimum(g("RO164_DES_flood"), g("RO099_ASC_flood"))
    ok = (dry > -32000) & (ma > -32000) & (js > -32000) & (oc > -32000) & (gu > -32000)
    V = int(__import__("os").environ.get("RULE", "5"))
    if V == 1:
        rule = ok & (dry < 30) & (mj >= 55) & (js >= 60) & (gu <= 250) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    elif V == 2:   # bare through April (planted after), green by Jul, dense Jul-Oct; late plantings allowed
        rule = ok & (dry < 20) & (ma < 40) & (js >= 60) & (oc >= 60) & (gu <= 290) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    elif V == 3:   # truly bare at some point Dec-Feb (< 0.20), dense Jul-Sep and Oct, green-up by mid Aug (late plantings allowed)
        rule = ok & (dry < 20) & (js >= 60) & (oc >= 60) & (gu <= 290) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    elif V == 4:   # v3 + some canopy by May-Jun (>= 0.35; rice is still bare then) + SAR canopy at season end (mean VH of both orbits >= -16 dB)
        rule = ok & (mj > -32000) & (dry < 20) & (mj >= 35) & (js >= 60) & (oc >= 60) & (gu <= 290) & (ev >= -1600) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    elif V == 6:   # strict: v5 + canopy by May-Jun >= 0.55 and green-up before mid Jun (<= 230 d): drops late plantings AND DSR rice look-alikes
        rule = ok & (mj > -32000) & (dry < 20) & (mj >= 55) & (js >= 60) & (oc >= 60) & (gu <= 230) & (ev >= -1600) & (ys >= 150) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    elif V == 7:   # v5 + no flooding: min VH Jun-Oct of either orbit > -20.5 dB (user polygons 15 Oct: early-transplanted rice added by v5)
        rule = ok & (mj > -32000) & (dry < 20) & (mj >= 35) & (js >= 60) & (oc >= 60) & (gu <= 290) & (ev >= -1600) & (ys >= 150) & (fl > -2050) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    else:   # v5 = v4 + a crop-like VH year (mean std of both orbits >= 1.5 dB): rule review 14 Oct: removes 12/12 forest blocks, keeps 99 % of cane
        rule = ok & (mj > -32000) & (dry < 20) & (mj >= 35) & (js >= 60) & (oc >= 60) & (gu <= 290) & (ev >= -1600) & (ys >= 150) & (octmin >= 30) & (octmin <= 100) & (m != 255)
    new = np.where(rule & (m == 0), 1, m).astype("uint8")
    src_b = np.where(m == 255, 255, np.where(m == 1, 1, np.where(rule, 2, 0))).astype("uint8")
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "cane_map_2026.tif", "w", **pr) as d: d.write(new, 1)
    with rasterio.open(o / "cane_source_2026.tif", "w", **pr) as d: d.write(src_b, 1)
    for f in ("cane_prob_2026.tif", "cane_prob_2026.qml", "oct_ndvi_min.tif"):
        if (s / f).exists(): shutil.copy(s / f, o / f)
    Q.paletted(o / "cane_map_2026.tif", Q.MAP_CLASSES)
    Q.paletted(o / "cane_source_2026.tif", {0: ("#e6e6e6", "non-cane"), 1: ("#33ef2a", "cane: model"), 2: ("#ff8c00", "cane: S2 plant-cane rule (added)")})
    res = {"site": site.name, "src": src, "cane_ac_model": round(float((m == 1).sum() * ACRE)), "cane_ac_added_by_rule": round(float((rule & (m == 0)).sum() * ACRE)),
           "cane_ac_total": round(float((new == 1).sum() * ACRE)), "rule": "v1: dry<0.30 & mayjun>=0.55 & julsep>=0.60 & greenup<=250 & oct>=0.30" if V == 1 else "v2: dry<0.20 & marapr<0.40 & julsep>=0.60 & octmed>=0.60 & greenup<=290 & octmin>=0.30" if V == 2 else "v3: dry<0.20 & julsep>=0.60 & octmed>=0.60 & greenup<=290 & octmin>=0.30" if V == 3 else "v4: v3 + mayjun>=0.35 & mean end_vh>=-16" if V == 4 else "v6: v5 strict (mayjun>=0.55, greenup<=230)" if V == 6 else "v7: v5 + no flood (min VH Jun-Oct > -20.5)" if V == 7 else "v5: v4 + mean VH year std>=1.5 (no forest)"}
    (o / "rule.json").write_text(json.dumps(res, indent=1)); print(res, flush=True); return res

if __name__ == "__main__":
    for p in sys.argv[3:]: one(p, sys.argv[1], sys.argv[2])

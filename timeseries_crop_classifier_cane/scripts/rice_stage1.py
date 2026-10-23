"""Rice stage 1 (user plan 16 Oct): transplanted rice from the S1 sowing / flooding dip, inside the layer2_final sugarcane class.
Rule (calibrated on user polygons + test-set blocks, sites/layer2/rice_sar_calib*.csv): min VH over the Jun-Aug 15-day
composites <= -20 dB on either orbit (cane flagged 3-4 %, rice 57-78 %). Output <site>/data/model/rice_stage1/
rice_sar_2026.tif (1 = rice by SAR dip, 0 = other sugarcane-class px, 255 = outside), vh_junaug_min.tif, stage1.json.
usage: python scripts/rice_stage1.py [site_dir ...]"""
import json, sys
from pathlib import Path
import numpy as np, rasterio
import site_paths as SP
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q
P = Path(__file__).resolve().parents[1]; JA = ["202606a", "202606b", "202607a", "202607b", "202608a", "202608b"]; THR = -20.0; ACRE = 0.0247105
def one(site):
    S = SP.site_dir(site) / "data"; o = S / "model/rice_stage1"; o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(S / "model/layer2_final/class3_2026.tif") as r: cls, prof = r.read(1), r.profile.copy()
    with rasterio.open(S / "features/s1_15d_2026nov.tif") as r:
        n = list(r.descriptions); mn = np.full(cls.shape, 99.0, "float32")
        for orb in ("RO164_DES", "RO099_ASC"):
            for p in JA:
                a = r.read(n.index(f"{orb}_VH_{p}") + 1).astype("float32"); a[a <= -32000] = np.nan; a /= 100
                mn = np.fmin(mn, np.where(np.isnan(a), 99, a))
    sug = cls == 1; rice = sug & (mn <= THR)
    out = np.where(cls == 65535, 255, np.where(rice, 1, 0)).astype("uint8"); out[(cls != 1) & (cls != 65535)] = 0
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "rice_sar_2026.tif", "w", **pr) as d: d.write(out, 1)
    with rasterio.open(o / "vh_junaug_min.tif", "w", **dict(pr, dtype="int16", nodata=-32768)) as d: d.write(np.where(mn > 90, -32768, np.round(mn * 100)).astype("int16"), 1)
    Q.paletted(o / "rice_sar_2026.tif", {0: ("#e6e6e6", "sugarcane class, no dip"), 1: ("#1B5E20", "rice (SAR sowing dip, stage 1)")})
    res = {"site": site, "sugarcane_class_ac": round(float(sug.sum() * ACRE)), "rice_stage1_ac": round(float(rice.sum() * ACRE)), "rule": f"min VH Jun-Aug <= {THR} dB (either orbit)"}
    (o / "stage1.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)
for s in (sys.argv[1:] or sorted(p.name for p in SP.all_sites("*_2026") if (p / "data/model/layer2_final").exists())): one(s)
print("STAGE1_DONE")

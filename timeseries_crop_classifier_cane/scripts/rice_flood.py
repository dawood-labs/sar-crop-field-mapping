"""Rice flood detector (user rule 16 Oct 2026): per-pixel RELATIVE dip on per-date Sentinel-1, consistent over consecutive
dates on both tracks, in VV and VH. No fixed dB threshold on one composite.
Per track: 3x3 mean (linear) per date, anomaly = value - pixel's own median over the season (VV and VH separately).
Both tracks merged in date order (one observation every ~6 days). An observation is "flooded" when the VH anomaly <= -DVH
and the VV anomaly <= -DVV. In the sowing window (WIN_LO..WIN_HI) we take the longest run of consecutive flooded
observations: flood_nobs, flood_days (first..last date of the run + 6), flood_start (doy), flood_depth (mean VH anomaly).
Rice candidate = flood_nobs >= MIN_OBS and flood_days >= MIN_DAYS (defaults 3 obs / 12 days).
Outputs <site>/data/model/rice_flood/: flood_nobs.tif, flood_days.tif, flood_start.tif, flood_depth.tif, nflood_total.tif,
rice_flood_2026.tif (1 = rice candidate inside the layer2_final sugarcane class, 2 = candidate outside it), flood.json.
usage: python scripts/rice_flood.py <site_dir> [more]      env DVH=4 DVV=3 MIN_OBS=3 MIN_DAYS=12 WIN_LO=0501 WIN_HI=0930
"""
import glob, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, rasterio
from scipy.ndimage import uniform_filter
import site_paths as SP
sys.path.insert(0, str(Path(__file__).resolve().parent)); import qgis_style as Q
P = Path(__file__).resolve().parents[1]; ACRE = 0.0247105
DVH, DVV = float(os.environ.get("DVH", 4)), float(os.environ.get("DVV", 3))
MIN_OBS, MIN_DAYS = int(os.environ.get("MIN_OBS", 3)), int(os.environ.get("MIN_DAYS", 12))
WIN_LO, WIN_HI = os.environ.get("WIN_LO", "0501"), os.environ.get("WIN_HI", "0930")


def load_track(tdir):
    files = sorted(glob.glob(f"{tdir}/*.tif")); dates = [re.search(r"(\d{8})", f).group(1) for f in files]
    def rd(f):
        with rasterio.open(f) as r: a = r.read().astype("float32")
        a[a <= -9000] = np.nan; lin = 10 ** (a / 10); ok = np.isfinite(lin)
        out = np.empty_like(a)
        for k in range(2):
            num = uniform_filter(np.where(ok[k], lin[k], 0), 3, mode="nearest"); den = uniform_filter(ok[k].astype("float32"), 3, mode="nearest")
            with np.errstate(all="ignore"): out[k] = 10 * np.log10(num / den)
        return out
    with ThreadPoolExecutor(8) as ex: arrs = list(ex.map(rd, files))
    vv = np.stack([a[0] for a in arrs]); vh = np.stack([a[1] for a in arrs]); del arrs
    with np.errstate(all="ignore"):
        vv -= np.nanmedian(vv, 0); vh -= np.nanmedian(vh, 0)            # anomaly vs the pixel's own median (per track)
    return dates, vv, vh


def one(site):
    t0 = time.time(); S = SP.site_dir(site) / "data"; o = S / "model/rice_flood"; o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(S / "model/layer2_final/class3_2026.tif") as r: cls, prof = r.read(1), r.profile.copy()
    obs = []                                                                # (date, flooded bool array)
    for tdir in sorted(glob.glob(str(S / "raw/s1/*"))):
        dates, vv, vh = load_track(tdir)
        for i, d in enumerate(dates):
            obs.append((d, (vh[i] <= -DVH) & (vv[i] <= -DVV)))
        del vv, vh; print(f"  {site} {Path(tdir).name}: {len(dates)} dates", flush=True)
    obs.sort(key=lambda x: x[0]); yr = obs[0][0][:4]
    H, W = cls.shape; run = np.zeros((H, W), "int16"); run_start = np.zeros((H, W), "int16"); best = np.zeros((H, W), "int16")
    best_days = np.zeros((H, W), "int16"); best_start = np.zeros((H, W), "int16"); best_depth = np.zeros((H, W), "float32")
    depth_sum = np.zeros((H, W), "float32"); ntot = np.zeros((H, W), "int16")
    import datetime as dt
    doy = lambda d: dt.date(int(d[:4]), int(d[4:6]), int(d[6:])).timetuple().tm_yday
    lo, hi = [d for d in dates if False], None
    for d, f in obs:
        if not (d[4:] >= WIN_LO and d[4:] <= WIN_HI and d[:4] == "2026"): continue
        day = doy(d); ntot += f
        run = np.where(f, run + 1, 0); run_start = np.where(f & (run == 1), day, run_start)
        days = np.where(f, day - run_start + 6, 0).astype("int16")
        better = f & (run > best)
        best = np.where(better, run, best); best_days = np.where(better, days, best_days); best_start = np.where(better, run_start, best_start)
    rice = (best >= MIN_OBS) & (best_days >= MIN_DAYS)
    out = np.where(cls == 65535, 255, np.where(rice & (cls == 1), 1, np.where(rice, 2, 0))).astype("uint8")
    pr = dict(prof, count=1, dtype="int16", nodata=-32768, compress="deflate", predictor=2)
    for name, a in (("flood_nobs", best), ("flood_days", best_days), ("flood_start", best_start), ("nflood_total", ntot)):
        with rasterio.open(o / f"{name}.tif", "w", **pr) as d: d.write(a.astype("int16"), 1)
    with rasterio.open(o / "rice_flood_2026.tif", "w", **dict(pr, dtype="uint8", nodata=255)) as d: d.write(out, 1)
    Q.paletted(o / "rice_flood_2026.tif", {0: ("#e6e6e6", "no flood run"), 1: ("#1B5E20", "rice candidate (in sugarcane class)"), 2: ("#80cbc4", "flood run outside sugarcane class")})
    res = {"site": site, "rule": f"VH anomaly <= -{DVH} & VV anomaly <= -{DVV} dB vs pixel median, run >= {MIN_OBS} obs & >= {MIN_DAYS} d in {WIN_LO}-{WIN_HI}",
           "sugarcane_class_ac": round(float((cls == 1).sum() * ACRE)), "rice_in_sugarcane_class_ac": round(float((out == 1).sum() * ACRE)),
           "flood_outside_class_ac": round(float((out == 2).sum() * ACRE)), "n_obs_window": int(sum(1 for d, _ in obs if d[4:] >= WIN_LO and d[4:] <= WIN_HI and d[:4] == "2026")), "sec": round(time.time() - t0)}
    (o / "flood.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)


for s in sys.argv[1:]: one(s)
print("FLOOD_DONE")

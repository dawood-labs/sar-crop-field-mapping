"""Per-pixel RELATIVE, run-based Sentinel-1 features from the per-date S1 files (user rule 16 Oct 2026: a pixel is judged
against its own values, consistently over consecutive dates on both tracks, in VV and VH; no fixed-dB single-date rule).
Per track: 3x3 mean (linear) per date, anomaly = value - the pixel's own median over all dates (VV, VH, gap = VV-VH).
Both tracks are merged in date order (one observation every ~6 days); runs = consecutive observations.
Output <site>/data/features/s1_runs_2026.tif (int16, scale noted per band):
 flood_run4      longest run of obs with VH anomaly <= -4 dB, 1 May-15 Oct               (obs)
 flood_run4_vv2  same with VV anomaly <= -2 dB as well                                    (obs)
 flood_days4_vv2 length of that run in days (first..last obs + 6)                         (days)
 flood_start4    start day-of-year of the longest flood_run4 run                          (doy)
 nflood4         number of obs with VH anomaly <= -4 dB, 1 May-15 Oct                      (obs)
 gap10_run       longest run of obs with VV-VH >= 10 dB, 1 Jul-15 Oct                     (obs)
 gapanom4_run    longest run of obs with gap anomaly >= +4 dB, 1 Jul-15 Oct               (obs)
 gap_sm_max      max of the 3-obs running mean of VV-VH, 1 Jul-15 Oct                      (dB x100)
 vv_sm_max       max of the 3-obs running mean of VV, 1 Aug-15 Oct                         (dB x100)
 vh_sm_min       min of the 3-obs running mean of VH, 1 May-15 Oct                         (dB x100)
 vh_sm_min_doy   doy of that minimum                                                        (doy)
 drylow4_run     longest run of obs with VH anomaly <= -4 dB, 1 Dec-30 Apr                (obs)
 greenup_doy     first doy after 1 Apr with VH anomaly >= +2 dB on 2 consecutive obs       (doy, 0 = none)
 vh_rise_may     VH 3-obs mean (15 May-15 Jun) minus VH 3-obs mean (1-30 Apr)              (dB x100)
 nobs            number of observations used
usage: python scripts/s1_runs.py <site_dir> [more]"""
import datetime as dt, glob, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
import site_paths as SP
P = Path(__file__).resolve().parents[1]; NB = 4
BANDS = ["flood_run4", "flood_run4_vv2", "flood_days4_vv2", "flood_start4", "nflood4", "gap10_run", "gapanom4_run", "gap_sm_max", "vv_sm_max",
         "vh_sm_min", "vh_sm_min_doy", "drylow4_run", "greenup_doy", "vh_rise_may", "nobs"]
doy = lambda d: dt.date(int(d[:4]), int(d[4:6]), int(d[6:])).timetuple().tm_yday
inwin = lambda d, lo, hi: d[:4] == "2026" and lo <= d[4:] <= hi


def read_track(files, win):
    def rd(f):
        with rasterio.open(f) as r: a = r.read(window=win).astype("float32")
        a[a <= -9000] = np.nan; lin = 10 ** (a / 10); o = np.empty_like(a)
        with np.errstate(all="ignore"):
            for k in range(2):
                o[k] = 10 * np.log10(uniform_filter(np.where(np.isfinite(lin[k]), lin[k], 0), 3, mode="nearest") / uniform_filter(np.isfinite(lin[k]).astype("float32"), 3, mode="nearest"))
        return o
    with ThreadPoolExecutor(8) as ex: arrs = list(ex.map(rd, files))
    vv = np.stack([a[0] for a in arrs]); vh = np.stack([a[1] for a in arrs]); del arrs
    with np.errstate(all="ignore"): avv = vv - np.nanmedian(vv, 0); avh = vh - np.nanmedian(vh, 0); gap = vv - vh; agap = gap - np.nanmedian(gap, 0)
    return vv, vh, gap, avv, avh, agap


def block(S, tracks, win):
    obs = []
    for tdir, files, dates in tracks:
        vv, vh, gap, avv, avh, agap = read_track(files, win)
        for i, d in enumerate(dates): obs.append((d, vv[i], vh[i], gap[i], avv[i], avh[i], agap[i]))
        del vv, vh, gap, avv, avh, agap
    obs.sort(key=lambda x: x[0]); shp = obs[0][1].shape; z = lambda t="int16": np.zeros(shp, t)
    run_f, best_f, start_f, best_start = z(), z(), z(), z(); run_fv, best_fv, start_fv, best_days = z(), z(), z(), z(); nfl = z()
    run_g, best_g, run_ga, best_ga = z(), z(), z(), z(); gap_sm_max = np.full(shp, -99.0, "float32"); vv_sm_max = np.full(shp, -99.0, "float32")
    vh_sm_min = np.full(shp, 99.0, "float32"); vh_sm_min_doy = z(); run_d, best_d = z(), z(); gup = z(); run_u = z(); nobs = z()
    apr, may = [], []; prev = []   # last 2 obs for 3-obs running means
    for d, vv, vh, gap, avv, avh, agap in obs:
        ok = np.isfinite(vh); nobs += ok; day = doy(d)
        if inwin(d, "0501", "1015"):
            f = ok & (avh <= -4); nfl += f
            run_f = np.where(f, run_f + 1, 0); start_f = np.where(f & (run_f == 1), day, start_f)
            b = f & (run_f > best_f); best_f = np.where(b, run_f, best_f); best_start = np.where(b, start_f, best_start)
            fv = f & (avv <= -2); run_fv = np.where(fv, run_fv + 1, 0); start_fv = np.where(fv & (run_fv == 1), day, start_fv)
            b = fv & (run_fv > best_fv); best_fv = np.where(b, run_fv, best_fv); best_days = np.where(b, day - start_fv + 6, best_days)
        if inwin(d, "0701", "1015"):
            g = ok & (gap >= 10); run_g = np.where(g, run_g + 1, 0); best_g = np.maximum(best_g, run_g)
            g = ok & (agap >= 4); run_ga = np.where(g, run_ga + 1, 0); best_ga = np.maximum(best_ga, run_ga)
        if inwin(d, "1201", "1231") or inwin(d, "0101", "0430") or d[:4] == "2025" and d[4:] >= "1201":
            f = ok & (avh <= -4); run_d = np.where(f, run_d + 1, 0); best_d = np.maximum(best_d, run_d)
        if inwin(d, "0401", "1015"):
            u = ok & (avh >= 2); run_u = np.where(u, run_u + 1, 0); gup = np.where((run_u == 2) & (gup == 0), day, gup)
        if len(prev) == 2:
            with np.errstate(all="ignore"):
                vv3 = np.nanmean([prev[0][0], prev[1][0], vv], 0); vh3 = np.nanmean([prev[0][1], prev[1][1], vh], 0); gap3 = np.nanmean([prev[0][2], prev[1][2], gap], 0)
            if inwin(d, "0701", "1015"): gap_sm_max = np.fmax(gap_sm_max, gap3)
            if inwin(d, "0801", "1015"): vv_sm_max = np.fmax(vv_sm_max, vv3)
            if inwin(d, "0501", "1015"):
                m = vh3 < vh_sm_min; vh_sm_min = np.where(m, vh3, vh_sm_min); vh_sm_min_doy = np.where(m, day, vh_sm_min_doy)
        if inwin(d, "0401", "0430"): apr.append(vh)
        if inwin(d, "0515", "0615"): may.append(vh)
        prev = (prev + [(vv, vh, gap)])[-2:]
    with np.errstate(all="ignore"): rise = np.nanmean(may, 0) - np.nanmean(apr, 0) if apr and may else np.zeros(shp, "float32")
    f100 = lambda a, fill: np.where(np.isfinite(a) & (np.abs(a) < 90), np.round(a * 100), fill).astype("int16")
    return np.stack([best_f, best_fv, best_days, best_start, nfl, best_g, best_ga, f100(gap_sm_max, -32768), f100(vv_sm_max, -32768), f100(vh_sm_min, -32768),
                     vh_sm_min_doy, best_d, gup, f100(rise, -32768), nobs]).astype("int16")


def one(site):
    t0 = time.time(); S = SP.site_dir(site) / "data"; out = S / "features/s1_runs_2026.tif"
    tracks = []
    for tdir in sorted(glob.glob(str(S / "raw/s1/*"))):
        files = sorted(glob.glob(f"{tdir}/*.tif")); tracks.append((tdir, files, [re.search(r"(\d{8})", f).group(1) for f in files]))
    with rasterio.open(tracks[0][1][0]) as r: H, W = r.shape; prof = r.profile.copy()
    prof.update(count=len(BANDS), dtype="int16", nodata=-32768, compress="deflate", predictor=2, tiled=False, BIGTIFF="IF_SAFER")
    edges = np.linspace(0, H, NB + 1).astype(int)
    with rasterio.open(out, "w", **prof) as d:
        for i, nm in enumerate(BANDS, 1): d.set_band_description(i, nm)
        for i in range(NB):
            r0, r1 = edges[i], edges[i + 1]; lo = max(r0 - 1, 0); hi = min(r1 + 1, H)      # 1-row halo for the 3x3 mean
            a = block(S, tracks, Window(0, lo, W, hi - lo))[:, r0 - lo: r0 - lo + (r1 - r0)]
            d.write(a, window=Window(0, r0, W, r1 - r0)); print(f"  {site} block {i + 1}/{NB} {time.time() - t0:.0f} s", flush=True)
    print(f"{site} -> {out} ({sum(len(t[2]) for t in tracks)} obs) {time.time() - t0:.0f} s", flush=True)


for s in sys.argv[1:]: one(s)
print("RUNS_DONE")

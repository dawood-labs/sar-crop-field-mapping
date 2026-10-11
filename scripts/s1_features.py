"""Sentinel-1 monthly features for the whole AOI grid: data/features/s1_monthly_w5.tif.

Per track (RO164_DES, RO099_ASC) and month (2026-03 .. 2026-09): mean of VV and VH in LINEAR power over the month's
acquisitions, after a 5 x 5 window mean (speckle), back to dB; plus VH-VV (dB). 7 months x 2 tracks x 3 = 42 bands,
band name = <track>_<var>_<YYYYMM>. Dates with >= 10 mm IMERG rain (AOI mean) in the 24 h before the acquisition are
dropped (wet surface changes backscatter), unless that would leave the month empty. October 2026 (1 date) is left out.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

PROJ = Path(__file__).resolve().parents[1]
ST = PROJ / "data/stacks"
# v1: 2026-03..2026-09 -> s1_monthly_w5.tif (user: current season only).
# v2: 2025-11..2026-09 -> s1_monthly_w5_v2.tif (audit showed cane/cassava green in Dec-Feb are mistaken for rice
#     without the months when the 2025 rice is already harvested). Usage: s1_features.py [START END OUTNAME]
_args = sys.argv[1:]
MONTHS = pd.period_range(_args[0] if _args else "2026-03", _args[1] if _args else "2026-09", freq="M")
OUT = PROJ / "data/features" / (_args[2] if len(_args) > 2 else "s1_monthly_w5.tif")
RAIN_MM, WIN = 10.0, 5


def track_var(track, var, rain):
    with rasterio.open(ST / f"S1_{track}_{var}.tif") as r:
        all_dates = pd.to_datetime(list(r.descriptions))
        keep = [i for i, d in enumerate(all_dates) if d.to_period("M") in set(MONTHS)]
        dates = all_dates[keep]
        a = r.read([i + 1 for i in keep]).astype("float32")        # only the months we need (memory)
    wet = set(rain[(rain.track == track) & (rain.imerg_24h >= RAIN_MM)].date)
    out = {}
    for m in MONTHS:
        idx = [i for i, d in enumerate(dates) if d.to_period("M") == m]
        dry = [i for i in idx if dates[i].strftime("%Y%m%d") not in wet]
        use = dry or idx
        lin = np.zeros(a.shape[1:], "float64"); cnt = np.zeros(a.shape[1:], "float64")
        for i in use:
            x = 10 ** (a[i] / 10)
            ok = np.isfinite(x)
            lin += uniform_filter(np.where(ok, x, 0), WIN, mode="nearest")
            cnt += uniform_filter(ok.astype("float64"), WIN, mode="nearest")
        with np.errstate(all="ignore"):
            out[f"{track}_{var}_{m.strftime('%Y%m')}"] = (10 * np.log10(lin / cnt)).astype("float32")
    return out, {str(m): (len([i for i, d in enumerate(dates) if d.to_period('M') == m]),
                          len([i for i, d in enumerate(dates) if d.to_period('M') == m and d.strftime('%Y%m%d') not in wet]))
                 for m in MONTHS}


def main():
    t0 = time.time()
    rain = pd.read_csv(PROJ / "data/rain/rain_s1_acquisitions.csv")
    rain["date"] = rain.date.astype(str)
    jobs = [(t, v) for t in ["RO164_DES", "RO099_ASC"] for v in ["VV", "VH"]]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sar_pipeline.resources import detect_resources
    res = detect_resources()
    workers = max(1, min(len(jobs), res.cpus))                     # each worker holds one track/var stack (~1 GB)
    print(f"{res.describe()} -> {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(lambda j: track_var(*j, rain), jobs))
    bands = {}
    for (t, v), (b, counts) in zip(jobs, res):
        bands.update(b)
        if v == "VH":
            print(t, "dates per month (all, kept after rain filter):", counts)
    names = []
    for t in ["RO164_DES", "RO099_ASC"]:
        for m in MONTHS:
            k = m.strftime("%Y%m")
            bands[f"{t}_VHmVV_{k}"] = bands[f"{t}_VH_{k}"] - bands[f"{t}_VV_{k}"]
        for v in ["VV", "VH", "VHmVV"]:
            names += [f"{t}_{v}_{m.strftime('%Y%m')}" for m in MONTHS]
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        prof = r.profile.copy()
    prof.update(count=len(names), dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, interleave="pixel", BIGTIFF="IF_SAFER")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT, "w", **prof) as dst:
        dst.write(np.stack([bands[n] for n in names]))
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)
        dst.update_tags(window=f"{WIN}x{WIN} linear mean", rain_filter=f"IMERG 24 h >= {RAIN_MM} mm dropped", months=f"{MONTHS[0]}..{MONTHS[-1]}")
    print(f"{len(names)} bands -> {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

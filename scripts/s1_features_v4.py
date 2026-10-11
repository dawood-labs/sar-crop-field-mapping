"""Sentinel-1 features v4: data/features/s1_features_v4.tif

Why (QC of rice v3 on 255 independent labels): almost all errors sit on field edges (8/8 false rice, 22/24 missed
rice; interior pixels 98 % correct) and the missed rice is mostly direct-seeded rice with no standing-water dip and a
weak canopy rise. v1-v3 used a 5x5 (50 m) window, which mixes the neighbouring field into edge pixels, and only
monthly means, which wash out a one- or two-date flooding dip.

Bands (per track RO164_DES and RO099_ASC; dB; dates with >= 10 mm IMERG rain in the previous 24 h dropped):
  <track>_<VV|VH|VHmVV>_<YYYYMM>  monthly mean in linear power, 3x3 window, Nov 2025 .. Sep 2026      (33 per track)
  <track>_VH1_<YYYYMM>            monthly VH of the pixel itself (1x1), Jun .. Sep 2026 (sharper edges)  (4)
  <track>_nwet_JunAug             number of dates with VH <= -20 dB in Jun-Aug (3x3)                     (1)
  <track>_vhmin_JunAug            lowest VH in Jun-Aug (3x3)                                             (1)
  <track>_vhrise                  mean VH 15 Aug-5 Oct minus mean VH 1 Jun-15 Jul (canopy growth, 3x3)  (1)
  <track>_vhlate                  mean VH 15 Sep-5 Oct (crop standing now, 3x3)                         (1)
  <track>_vhdry_trend             mean VH Apr-May minus mean VH Feb-Mar (dry-season regrowth = cane)    (1)
Usage: s1_features_v4.py  (memory: one track x var stack in memory at a time per worker; sized by safe_resources)
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
from safe_resources import workers  # noqa: E402

ST, OUT = PROJ / "data/stacks", PROJ / "data/features/s1_features_v4.tif"
MONTHS = pd.period_range("2025-11", "2026-09", freq="M")
RAIN_MM, WIN = 10.0, 3


def load(track, var, rain):
    with rasterio.open(ST / f"S1_{track}_{var}.tif") as r:
        dates = pd.to_datetime(list(r.descriptions))
        keep = [i for i, d in enumerate(dates) if pd.Timestamp("2025-11-01") <= d <= pd.Timestamp("2026-10-05")]
        a = r.read([i + 1 for i in keep]).astype("float32")
    dates = dates[keep]
    wet = set(rain[(rain.track == track) & (rain.imerg_24h >= RAIN_MM)].date)
    dry = np.array([d.strftime("%Y%m%d") not in wet for d in dates])
    lin = 10 ** (a / 10)
    ok = np.isfinite(lin)
    num = np.stack([uniform_filter(np.where(ok[i], lin[i], 0), WIN, mode="nearest") for i in range(len(dates))])
    den = np.stack([uniform_filter(ok[i].astype("float32"), WIN, mode="nearest") for i in range(len(dates))])
    with np.errstate(all="ignore"):
        w3 = (10 * np.log10(num / den)).astype("float32")
    return dates, dry, a, w3


def monthly(dates, dry, x, m):
    idx = [i for i, d in enumerate(dates) if d.to_period("M") == m]
    use = [i for i in idx if dry[i]] or idx
    with np.errstate(all="ignore"):
        return (10 * np.log10(np.nanmean(10 ** (x[use] / 10), 0))).astype("float32")


def window(dates, dry, x, a, b, fn):
    idx = [i for i, d in enumerate(dates) if pd.Timestamp(a) <= d <= pd.Timestamp(b) and dry[i]]
    with np.errstate(all="ignore"):
        return fn(x[idx]).astype("float32")


def track_bands(track, rain):
    out = {}
    for var in ["VV", "VH"]:
        dates, dry, raw, w3 = load(track, var, rain)
        for m in MONTHS:
            out[f"{track}_{var}_{m.strftime('%Y%m')}"] = monthly(dates, dry, w3, m)
        if var == "VH":
            for m in pd.period_range("2026-06", "2026-09", freq="M"):
                out[f"{track}_VH1_{m.strftime('%Y%m')}"] = monthly(dates, dry, raw, m)
            out[f"{track}_nwet_JunAug"] = window(dates, np.ones_like(dry), w3, "2026-06-01", "2026-08-31", lambda q: (q <= -20).sum(0))
            out[f"{track}_vhmin_JunAug"] = window(dates, np.ones_like(dry), w3, "2026-06-01", "2026-08-31", lambda q: np.nanmin(q, 0))
            late = window(dates, dry, w3, "2026-08-15", "2026-10-05", lambda q: np.nanmean(q, 0))
            early = window(dates, dry, w3, "2026-06-01", "2026-07-15", lambda q: np.nanmean(q, 0))
            out[f"{track}_vhrise"] = late - early
            out[f"{track}_vhlate"] = window(dates, dry, w3, "2026-09-15", "2026-10-05", lambda q: np.nanmean(q, 0))
            out[f"{track}_vhdry_trend"] = (window(dates, dry, w3, "2026-04-01", "2026-05-31", lambda q: np.nanmean(q, 0))
                                           - window(dates, dry, w3, "2026-02-01", "2026-03-31", lambda q: np.nanmean(q, 0)))
        del raw, w3
    for m in MONTHS:
        k = m.strftime("%Y%m")
        out[f"{track}_VHmVV_{k}"] = out[f"{track}_VH_{k}"] - out[f"{track}_VV_{k}"]
    return out


def main():
    t0 = time.time()
    rain = pd.read_csv(PROJ / "data/rain/rain_s1_acquisitions.csv"); rain["date"] = rain.date.astype(str)
    tracks = ["RO164_DES", "RO099_ASC"]
    n = workers(8 * 2**30, cap=len(tracks))           # a track holds ~2 stacks + 3x3 copies (~6-8 GB)
    print(f"{n} workers", flush=True)
    with ThreadPoolExecutor(n) as pool:
        res = list(pool.map(lambda t: track_bands(t, rain), tracks))
    bands = {k: v for r in res for k, v in r.items()}
    names = sorted(bands)
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        prof = r.profile.copy()
    prof.update(count=len(names), dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256, interleave="pixel", BIGTIFF="IF_SAFER")
    with rasterio.open(OUT, "w", **prof) as dst:
        dst.write(np.stack([bands[k] for k in names]))
        for i, k in enumerate(names, 1):
            dst.set_band_description(i, k)
    print(f"{len(names)} bands -> {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

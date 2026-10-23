"""Sentinel-1 15-day composites for the standing-cane model: data/features/s1_15d_<season>.tif

Why 15-day composites and not single dates: 2025 had only Sentinel-1A (12-day repeat per track), 2026 has S1C / S1D
(S1A ended June 2026), so the number and timing of dates differ between the years. Half-month means put both years
on the same time axis, so a model trained on 2025 can be applied to 2026.
Each band: mean of the dates in that half-month (1-15, 16-end) in LINEAR power, after a 3x3 window mean (speckle),
back to dB. VH-VV ratio band = VH dB - VV dB of the composite. A half-month without any date is filled from the
neighbouring half-months (linear interpolation in time) and counted in data/features/s1_15d_<season>_gaps.csv.
Season window (both years end on the same day of year, 10 Oct, because 2026 data ends there):
  2025: 2024-09-01 .. 2025-10-10        2026: 2025-09-01 .. 2026-10-10
Bands: <track>_<VV|VH|VHmVV>_<YYYYMM><a|b>   (a = days 1-15, b = 16-end)
Stored as int16 = dB x 100 (nodata -32768): half the size of float32, 0.01 dB precision (storage matters here).
Processed in row blocks (with a 1-row halo for the 3x3 window) so memory stays small.
usage: python scripts/s1_composites.py 2025
"""
from __future__ import annotations

import os
import site_paths as SP
os.environ.setdefault("GDAL_CACHEMAX", "256")

import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

PROJ = SP.site_dir(os.environ["SC_ROOT"]) if os.environ.get("SC_ROOT") else Path(__file__).resolve().parents[1]   # SC_ROOT = another site (external test)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_resources import workers  # noqa: E402

SEASONS = {"2025": ("2024-09-01", "2025-10-10"), "2026": ("2025-09-01", "2026-10-10"),
           "2025jan": ("2025-01-01", "2025-10-10"), "2026jan": ("2026-01-01", "2026-10-10"),   # model window (user: Jan start)
           "2025nov": ("2024-11-01", "2025-10-10"), "2026nov": ("2025-11-01", "2026-10-10")}   # Nov start test (13 Oct)
# values below these are border / no-data noise, not backscatter (a -47.9 dB VV date bent a composite by -3 dB)
MIN_DB = {"VV": float(os.environ.get("S1_MIN_VV", "-30")), "VH": float(os.environ.get("S1_MIN_VH", "-40"))}
# extra stack folders with more dates of the same grid (e.g. Nov-Dec downloaded later), comma separated
EXTRA = [Path(x) for x in os.environ.get("S1_EXTRA_STACKS", "").split(",") if x]
TRACKS = os.environ.get("S1_TRACKS", "RO164_DES,RO099_ASC").split(",")          # stack folder names of this site
# band names in the output use the model's track names (districts outside the RO164 / RO099 footprint, 14 Oct 2026):
# S1_TRACK_ALIAS="RO091_DES:RO164_DES,RO026_ASC:RO099_ASC" -> features of another descending / ascending track are
# written under the model's column names (same pass direction; incidence angles differ by a few degrees).
ALIAS = dict(x.split(":") for x in os.environ.get("S1_TRACK_ALIAS", "").split(",") if x)
BLOCK = 256


def periods(start, end):
    out = []
    for m in pd.period_range(start[:7], end[:7], freq="M"):
        a0, mid, b1 = m.start_time, m.start_time + pd.Timedelta(days=15), m.end_time.normalize() + pd.Timedelta(days=1)
        for tag, lo, hi in (("a", a0, mid), ("b", mid, b1)):
            lo, hi = max(lo, pd.Timestamp(start)), min(hi, pd.Timestamp(end) + pd.Timedelta(days=1))
            if lo < hi:
                out.append((f"{m.strftime('%Y%m')}{tag}", lo, hi))
    return out


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2025"
    start, end = SEASONS[season]
    t0 = time.time()
    P = periods(start, end)
    out = PROJ / f"data/features/s1_15d_{season}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = {}
    for t in TRACKS:
        for v in ("VV", "VH"):
            files = [PROJ / f"data/stacks/S1_{t}_{v}.tif"] + [e / f"S1_{t}_{v}.tif" for e in EXTRA if (e / f"S1_{t}_{v}.tif").exists()]
            parts = [(f, pd.to_datetime(list(rasterio.open(f).descriptions))) for f in files]
            r = rasterio.open(files[0])
            d = pd.DatetimeIndex(np.concatenate([q[1].values for q in parts]))
            src[(t, v)] = (r, d, parts)
    H, W = src[(TRACKS[0], "VV")][0].shape
    names, gaps = [], []
    for t in TRACKS:
        d = src[(t, "VV")][1]
        for v in ("VV", "VH", "VHmVV"):
            for tag, lo, hi in P:
                names.append(f"{ALIAS.get(t, t)}_{v}_{tag}")
        for tag, lo, hi in P:
            gaps.append({"track": t, "period": tag, "n_dates": int(((d >= lo) & (d < hi)).sum())})
    pd.DataFrame(gaps).to_csv(PROJ / f"data/features/s1_15d_{season}_gaps.csv", index=False)
    prof = dict(src[(TRACKS[0], "VV")][0].profile, count=len(names), dtype="int16", nodata=-32768, compress="deflate",
                predictor=2, tiled=True, blockxsize=256, blockysize=256, interleave="band", BIGTIFF="YES")
    dst = rasterio.open(out, "w", **prof)
    for i, n in enumerate(names, 1):
        dst.set_band_description(i, n)

    def block(r0):
        r1 = min(r0 + BLOCK, H); h0, h1 = max(r0 - 1, 0), min(r1 + 1, H)
        res = {}
        for t in TRACKS:
            comp = {}
            for v in ("VV", "VH"):
                arrs, dts = [], []
                for f, dq in src[(t, v)][2]:
                    keep = [i for i, x in enumerate(dq) if pd.Timestamp(start) <= x <= pd.Timestamp(end) + pd.Timedelta(days=1)]
                    if not keep:
                        continue
                    r = rasterio.open(f)   # own handle per thread (not thread-safe)
                    arrs.append(r.read([i + 1 for i in keep], window=Window(0, h0, W, h1 - h0)).astype("float32")); r.close()
                    dts.append(dq[keep].values)
                a = np.concatenate(arrs); dd = pd.DatetimeIndex(np.concatenate(dts))
                o = np.argsort(dd.values); a, dd = a[o], dd[o]
                a[a < MIN_DB[v]] = np.nan
                lin = 10 ** (a / 10); ok = np.isfinite(lin)
                num = np.stack([uniform_filter(np.where(ok[k], lin[k], 0), 3, mode="nearest") for k in range(len(dd))])
                den = np.stack([uniform_filter(ok[k].astype("float32"), 3, mode="nearest") for k in range(len(dd))])
                with np.errstate(all="ignore"):
                    w3 = num / den
                cs = np.full((len(P), h1 - h0, W), np.nan, "float32")
                for j, (tag, lo, hi) in enumerate(P):
                    sel = (dd >= lo) & (dd < hi)
                    if sel.any():
                        with np.errstate(all="ignore"):
                            cs[j] = np.nanmean(w3[sel], 0)
                # fill empty half-months by linear interpolation along time (per pixel)
                idx = np.arange(len(P)); have = ~np.isnan(cs).all(axis=(1, 2))
                if (~have).any() and have.sum() >= 2:
                    for j in np.nonzero(~have)[0]:
                        lo_j = idx[have & (idx < j)]; hi_j = idx[have & (idx > j)]
                        if len(lo_j) and len(hi_j):
                            a_, b_ = lo_j[-1], hi_j[0]; w = (j - a_) / (b_ - a_)
                            cs[j] = (1 - w) * cs[a_] + w * cs[b_]
                        else:
                            cs[j] = cs[lo_j[-1]] if len(lo_j) else cs[hi_j[0]]
                with np.errstate(all="ignore"):
                    comp[v] = (10 * np.log10(cs))[:, r0 - h0:r0 - h0 + (r1 - r0)]
            comp["VHmVV"] = comp["VH"] - comp["VV"]
            for v in ("VV", "VH", "VHmVV"):
                res[(t, v)] = comp[v]
        return r0, r1, res

    n = workers(int(1.5 * 2**30))   # measured < 1 GB per 256-row block (20 Oct 2026: 3 -> 8 workers on the 8-core pod)
    print(f"{len(names)} bands, {len(P)} half-months, {n} workers", flush=True)
    rows = list(range(0, H, BLOCK))
    with ThreadPoolExecutor(n) as pool:
        for k, (r0, r1, res) in enumerate(pool.map(block, rows), 1):
            i = 1
            for t in TRACKS:
                for v in ("VV", "VH", "VHmVV"):
                    for j in range(len(P)):
                        a = res[(t, v)][j]
                        dst.write(np.where(np.isfinite(a), np.round(a * 100), -32768).astype("int16"), i,
                                  window=Window(0, r0, W, r1 - r0)); i += 1
            if k % 4 == 0 or k == len(rows):
                print(f"  rows {r1}/{H} ({time.time() - t0:.0f} s)", flush=True)
    dst.close()
    print(f"-> {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

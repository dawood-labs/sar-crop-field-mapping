"""Date-free phenology features for the standing-cane model (round 3).

Why: the round-2 model only had calendar-bound features (15-day S1 bins, S2 Dec-Apr). Cane planted late (Feb-May) on
land bare in the dry season was missed, and cane fields turned to rice were kept. These features describe the crop
cycle instead of the calendar, and add the S2 wet season "whenever a clear date exists".

SAR (per orbit, from <site>/data/features/s1_15d_<year>nov.tif VH bands, 3-composite rolling median first):
  last_cut   index of the last composite with VH <= -19 dB (harvest / bare), -1 = none
  grow_len   consecutive composites with VH >= -16.5 dB ending at the last composite (canopy persistence)
  grow_total composites with VH >= -16.5 after the last cut
  peak_vh    max VH after the last cut (dB)
  end_vh     median VH of the last 4 composites
  flood      min VH over the second half of the series (<= -21 = flooded paddy)
  dip        min VH over the series
  ystd       std of VH over the series (forest / orchard small)
S2 (from the stacks, Cloud Score+ cs_cdf >= 60; NaN where no clear date):
  ndvi_dry_min (Dec-Feb), ndvi_marapr_max, ndvi_mayjun_max, ndvi_julsep_med, ndvi_oct_med (from 15 Sep),
  ndvi_last90_max (clear dates in the last 90 days), ndvi_max, greenup_idx (first clear date after 1 Feb with
  NDVI >= 0.5, as days since 1 Nov), n_clear_mayoct
Output: <site>/data/features/pheno_<year>.tif, int16 x 100 (dB x 100, NDVI x 100, counts / days as is), -32768 no data.
usage: python scripts/pheno_features.py <year> sites/<site> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

ND = -32768
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def fill_nan(a):
    """Forward then backward fill along axis 0 (time)."""
    out = a.copy()
    for t in range(1, out.shape[0]):
        m = np.isnan(out[t]); out[t][m] = out[t - 1][m]
    for t in range(out.shape[0] - 2, -1, -1):
        m = np.isnan(out[t]); out[t][m] = out[t + 1][m]
    return out


def sar_metrics(vh):
    """vh: (T, N) float dB with NaN. Returns dict of (N,) arrays."""
    T, N = vh.shape
    v = fill_nan(vh)
    allnan = np.isnan(v[0])
    # rolling median of 3 (edges: 2 values)
    sm = v.copy()
    sm[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
    cut = sm <= -19
    idx = np.arange(T)[:, None]
    last_cut = np.where(cut.any(0), (idx * cut).max(0), -1).astype("float32")
    hi = sm >= -16.5
    # consecutive hi ending at the last composite
    grow_len = np.zeros(N, "float32"); alive = np.ones(N, bool)
    for t in range(T - 1, -1, -1):
        alive &= hi[t]; grow_len += alive
    after = idx > last_cut[None, :]
    grow_total = (hi & after).sum(0).astype("float32")
    peak = np.where(after, sm, -np.inf).max(0).astype("float32")
    end = np.median(sm[-4:], axis=0).astype("float32")
    flood = sm[T // 2:].min(0).astype("float32")
    dip = sm.min(0).astype("float32")
    ystd = sm.std(0).astype("float32")
    out = dict(last_cut=last_cut, grow_len=grow_len, grow_total=grow_total, peak_vh=peak, end_vh=end, flood=flood, dip=dip, ystd=ystd)
    for k in out:
        out[k][allnan] = np.nan
    return out


def sar_block(site, year):
    f = site / "data/features" / f"s1_15d_{year}nov.tif"
    with rasterio.open(f) as r:
        names = list(r.descriptions); prof = r.profile.copy(); H, W = r.height, r.width
        res, cols = [], []
        for orb in ("RO164_DES", "RO099_ASC"):
            bands = [i + 1 for i, n in enumerate(names) if n.startswith(f"{orb}_VH_")]
            vh = r.read(bands).astype("float32").reshape(len(bands), -1)
            vh[vh <= -32000] = np.nan
            vh /= 100.0
            m = sar_metrics(vh)
            del vh
            for k, v in m.items():
                res.append(v); cols.append(f"{orb}_{k}")
            log(f"SAR {orb}: {len(bands)} composites")
    return res, cols, prof, (H, W)


def s2_block(site, year, shape):
    st = site / "data/stacks"
    with rasterio.open(st / "S2_cs_cdf.tif") as r:
        dates = pd.to_datetime(list(r.descriptions))
    H, W = shape
    def ndvi(i):
        with rasterio.open(st / "S2_cs_cdf.tif") as rc, rasterio.open(st / "S2_B4.tif") as r4, rasterio.open(st / "S2_B8.tif") as r8:
            cs = rc.read(i + 1); b4 = r4.read(i + 1).astype("float32"); b8 = r8.read(i + 1).astype("float32")
        ok = (cs >= 60) & (cs <= 100) & (b8 > 0)
        n = (b8 - b4) / np.maximum(b8 + b4, 1)
        n[~ok] = np.nan
        return n.ravel()
    t0 = pd.Timestamp(f"{year - 1}-11-01")
    win = {"dry": (f"{year - 1}-12-01", f"{year}-02-28"), "marapr": (f"{year}-03-01", f"{year}-04-30"),
           "mayjun": (f"{year}-05-01", f"{year}-06-30"), "julsep": (f"{year}-07-01", f"{year}-09-30"),
           "oct": (f"{year}-09-15", f"{year}-12-31")}
    last = dates.max() - pd.Timedelta(days=90)
    N = H * W
    mx = np.full(N, np.nan, "float32"); l90 = np.full(N, np.nan, "float32"); green = np.full(N, np.nan, "float32")
    nclear = np.zeros(N, "float32")
    stats = {k: [] for k in win}
    with ThreadPoolExecutor(4) as ex:
        for i, (d, n) in enumerate(zip(dates, ex.map(ndvi, range(len(dates))))):
            mx = np.fmax(mx, n)
            if d >= last:
                l90 = np.fmax(l90, n)
            if d >= pd.Timestamp(f"{year}-05-01"):
                nclear += ~np.isnan(n)
            if d >= pd.Timestamp(f"{year}-02-01"):
                g = np.isnan(green) & (n >= 0.5)
                green[g] = (d - t0).days
            for k, (a, b) in win.items():
                if pd.Timestamp(a) <= d <= pd.Timestamp(b):
                    stats[k].append(n)
            if i % 10 == 0:
                log(f"S2 {i + 1}/{len(dates)} {d.date()}")
    def agg(k, fn):
        s = stats[k]
        return fn(np.stack(s), axis=0) if s else np.full(N, np.nan, "float32")
    with np.errstate(all="ignore"):
        res = [agg("dry", np.nanmin), agg("marapr", np.nanmax), agg("mayjun", np.nanmax), agg("julsep", np.nanmedian),
               agg("oct", np.nanmedian), l90, mx, green, nclear]
    cols = ["ndvi_dry_min", "ndvi_marapr_max", "ndvi_mayjun_max", "ndvi_julsep_med", "ndvi_oct_med", "ndvi_last90_max",
            "ndvi_max", "greenup_days", "n_clear_mayoct"]
    return res, cols


def one(site, year):
    site = Path(site).resolve()
    res, cols, prof, shape = sar_block(site, year)
    r2, c2 = s2_block(site, year, shape)
    res += r2; cols += c2
    out = site / "data/features" / f"pheno_{year}.tif"
    prof.update(count=len(cols), dtype="int16", nodata=ND, compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out, "w", **prof) as dst:
        for b, (a, c) in enumerate(zip(res, cols), 1):
            scale = 1 if c.split("_", 2)[-1] in ("last_cut", "grow_len", "grow_total") or c in ("greenup_days", "n_clear_mayoct") else 100
            v = np.where(np.isfinite(a), np.clip(np.round(a * scale), -32000, 32000), ND).astype("int16")
            dst.write(v.reshape(shape), b); dst.set_band_description(b, c)
    log("wrote", out, len(cols), "bands")


if __name__ == "__main__":
    year = int(sys.argv[1])
    for s in sys.argv[2:]:
        one(s, year)

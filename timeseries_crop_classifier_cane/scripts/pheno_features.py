"""Date-free phenology features (round 3) - fast block-parallel version (20 Oct 2026; v1 kept as pheno_features_v1.py).
Date-free phenology features (same definitions and arithmetic as pheno_features_v1.py) computed in ROW BLOCKS on all
cores. Every feature is a per-pixel function of that pixel's time series, so block-wise processing gives bit-identical output;
v1 read whole bands date by date on one core (10+ min for a 64k-acre district), this reads one row block of every date per
worker (8 workers on the 8-core pod -> ~1 min).

SAR (per orbit, from <site>/data/features/s1_15d_<year>nov.tif VH bands, 3-composite rolling median first):
  last_cut, grow_len, grow_total, peak_vh, end_vh, flood, dip, ystd   (see pheno_features.py)
S2 (stacks B4, B8, cs_cdf; Cloud Score+ cs_cdf >= 60): ndvi_dry_min, ndvi_marapr_max, ndvi_mayjun_max, ndvi_julsep_med,
  ndvi_oct_med, ndvi_last90_max, ndvi_max, greenup_days, n_clear_mayoct
Output: <site>/data/features/pheno_<year>.tif, int16 x 100 (dB x 100, NDVI x 100, counts / days as is), -32768 no data.
usage: python scripts/pheno_features.py <year> sites/<site> [...]        env OUT=<file> (default pheno_<year>.tif), ROWS=128
"""
from __future__ import annotations

import os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from pheno_features_v1 import sar_metrics  # noqa: E402  (identical per-pixel maths)
import site_paths as SP

ND = -32768
T0 = time.time()
SAR_COLS = ["last_cut", "grow_len", "grow_total", "peak_vh", "end_vh", "flood", "dip", "ystd"]
S2_COLS = ["ndvi_dry_min", "ndvi_marapr_max", "ndvi_mayjun_max", "ndvi_julsep_med", "ndvi_oct_med", "ndvi_last90_max", "ndvi_max", "greenup_days", "n_clear_mayoct"]
UNSCALED = {"last_cut", "grow_len", "grow_total", "greenup_days", "n_clear_mayoct"}


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def block(args):
    """One row block: returns (r0, int16 array (bands, rows, W)) with all SAR + S2 features, scaled like v1."""
    site, year, r0, r1 = args
    site = Path(site); st = site / "data/stacks"; win = Window(0, r0, None, r1 - r0)
    out = []
    with rasterio.open(site / "data/features" / f"s1_15d_{year}nov.tif") as r:
        names = list(r.descriptions); W = r.width; win = Window(0, r0, W, r1 - r0)
        for orb in ("RO164_DES", "RO099_ASC"):
            bands = [i + 1 for i, n in enumerate(names) if n.startswith(f"{orb}_VH_")]
            vh = r.read(bands, window=win).astype("float32").reshape(len(bands), -1)
            vh[vh <= -32000] = np.nan
            vh /= 100.0
            m = sar_metrics(vh)
            out += [m[k] for k in SAR_COLS]
    with rasterio.open(st / "S2_cs_cdf.tif") as rc, rasterio.open(st / "S2_B4.tif") as r4, rasterio.open(st / "S2_B8.tif") as r8:
        dates = pd.to_datetime(list(rc.descriptions))
        cs = rc.read(window=win); b4 = r4.read(window=win).astype("float32"); b8 = r8.read(window=win).astype("float32")
    T = len(dates); N = (r1 - r0) * W
    cs = cs.reshape(T, N); b4 = b4.reshape(T, N); b8 = b8.reshape(T, N)
    t0 = pd.Timestamp(f"{year - 1}-11-01")
    win_ = {"dry": (f"{year - 1}-12-01", f"{year}-02-28"), "marapr": (f"{year}-03-01", f"{year}-04-30"),
            "mayjun": (f"{year}-05-01", f"{year}-06-30"), "julsep": (f"{year}-07-01", f"{year}-09-30"),
            "oct": (f"{year}-09-15", f"{year}-12-31")}
    last = dates.max() - pd.Timedelta(days=90)
    mx = np.full(N, np.nan, "float32"); l90 = np.full(N, np.nan, "float32"); green = np.full(N, np.nan, "float32")
    nclear = np.zeros(N, "float32"); stats = {k: [] for k in win_}
    for i, d in enumerate(dates):                      # same date order and same update order as v1
        ok = (cs[i] >= 60) & (cs[i] <= 100) & (b8[i] > 0)
        n = (b8[i] - b4[i]) / np.maximum(b8[i] + b4[i], 1)
        n[~ok] = np.nan
        mx = np.fmax(mx, n)
        if d >= last:
            l90 = np.fmax(l90, n)
        if d >= pd.Timestamp(f"{year}-05-01"):
            nclear += ~np.isnan(n)
        if d >= pd.Timestamp(f"{year}-02-01"):
            g = np.isnan(green) & (n >= 0.5)
            green[g] = (d - t0).days
        for k, (a, b) in win_.items():
            if pd.Timestamp(a) <= d <= pd.Timestamp(b):
                stats[k].append(n)
    def agg(k, fn):
        s = stats[k]
        return fn(np.stack(s), axis=0) if s else np.full(N, np.nan, "float32")
    with np.errstate(all="ignore"):
        out += [agg("dry", np.nanmin), agg("marapr", np.nanmax), agg("mayjun", np.nanmax), agg("julsep", np.nanmedian),
                agg("oct", np.nanmedian), l90, mx, green, nclear]
    cols = [f"{orb}_{k}" for orb in ("RO164_DES", "RO099_ASC") for k in SAR_COLS] + S2_COLS
    res = np.empty((len(out), r1 - r0, W), "int16")
    for b, (a, c) in enumerate(zip(out, cols)):
        scale = 1 if c.split("_", 2)[-1] in UNSCALED or c in UNSCALED else 100
        res[b] = np.where(np.isfinite(a), np.clip(np.round(a * scale), -32000, 32000), ND).astype("int16").reshape(r1 - r0, W)
    return r0, res


def one(site, year):
    site = SP.site_dir(site)
    with rasterio.open(site / "data/features" / f"s1_15d_{year}nov.tif") as r:
        prof = r.profile.copy(); H, W = r.height, r.width
    cols = [f"{orb}_{k}" for orb in ("RO164_DES", "RO099_ASC") for k in SAR_COLS] + S2_COLS
    out = site / "data/features" / os.environ.get("OUT", f"pheno_{year}.tif")
    prof.update(count=len(cols), dtype="int16", nodata=ND, compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256)
    rows = int(os.environ.get("ROWS", 128)); jobs = [(str(site), year, r0, min(r0 + rows, H)) for r0 in range(0, H, rows)]
    n = min(safe_resources.workers(1 * 2**30), len(jobs))     # 128-row block of 60 dates x 3 bands < 1 GB
    with rasterio.open(out, "w", **prof) as dst, ProcessPoolExecutor(n) as ex:
        for b, c in enumerate(cols, 1):
            dst.set_band_description(b, c)
        for k, (r0, res) in enumerate(ex.map(block, jobs), 1):
            dst.write(res, window=Window(0, r0, W, res.shape[1]))
            if k % 8 == 0 or k == len(jobs):
                log(f"{site.name} rows {min(r0 + rows, H)}/{H}")
    log("wrote", out, len(cols), "bands", f"({n} workers)")


if __name__ == "__main__":
    year = int(sys.argv[1])
    for s in sys.argv[2:]:
        one(s, year)

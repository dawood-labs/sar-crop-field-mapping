"""Dry-season Sentinel-2 features for the S2 test (user, 13 Oct 2026: "test first if it benefits").
Per month Dec (year-1) .. Apr: median of NDVI = (B8-B4)/(B8+B4) and NDRE = (B8-B5)/(B8+B5) over cloud-free dates
(Cloud Score+ cs_cdf >= 0.60); no clear date that month -> NaN (LightGBM handles missing values).
Rice lies bare Jan-Apr, ratoon cane regrows from Feb-Apr, cassava differs in timing: optical should help there.
Output: <site>/data/features/s2_dry_<year>.tif (10 bands float32 x 1000 as int16, -32768 = no data)
20 Oct 2026: computed in row blocks on all cores (per-pixel maths unchanged, output bit-identical to the v1 whole-band version).
usage: python scripts/s2_dry_features.py <year> sites/<site> [sites/<site> ...]      env OUT=<file>, ROWS=256
"""
from __future__ import annotations

import os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402
import site_paths as SP


def months_idx(dates, year):
    months = [f"{year - 1}-12"] + [f"{year}-{m:02d}" for m in range(1, 5)]
    return [(m, [i + 1 for i, d in enumerate(dates) if d.strftime("%Y-%m") == m]) for m in months]


def block(args):
    st, year, r0, r1, W = args
    st = Path(st); win = Window(0, r0, W, r1 - r0); out = []
    with rasterio.open(st / "S2_B8.tif") as r8, rasterio.open(st / "S2_B4.tif") as r4, rasterio.open(st / "S2_B5.tif") as r5, rasterio.open(st / "S2_cs_cdf.tif") as rc:
        dates = pd.to_datetime(list(r8.descriptions))
        for m, idx in months_idx(dates, year):
            if not idx:
                out += [np.full((r1 - r0, W), -32768, "int16")] * 2; continue
            b8 = r8.read(idx, window=win).astype("float32"); b4 = r4.read(idx, window=win).astype("float32"); b5 = r5.read(idx, window=win).astype("float32"); cs = rc.read(idx, window=win)
            ok = (cs >= 60) & (cs <= 100) & (b8 > 0)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ndvi = np.where(ok, (b8 - b4) / (b8 + b4), np.nan); ndre = np.where(ok, (b8 - b5) / (b8 + b5), np.nan)
                arrs = [np.nanmedian(a, 0) for a in (ndvi, ndre)]
            out += [np.where(np.isfinite(a), np.round(a * 1000), -32768).astype("int16") for a in arrs]
    return r0, np.stack(out)


def one(site: Path, year: int):
    st = site / "data/stacks"
    with rasterio.open(st / "S2_B8.tif") as r:
        dates = pd.to_datetime(list(r.descriptions)); prof = r.profile.copy(); H, W = r.height, r.width
    names = [n for m, _ in months_idx(dates, year) for n in (f"S2_NDVI_{m.replace('-', '')}", f"S2_NDRE_{m.replace('-', '')}")]
    p = dict(prof, count=len(names), dtype="int16", nodata=-32768, compress="deflate", predictor=2)
    f = site / "data/features" / os.environ.get("OUT", f"s2_dry_{year}.tif")
    rows = int(os.environ.get("ROWS", 256)); jobs = [(str(st), year, r0, min(r0 + rows, H), W) for r0 in range(0, H, rows)]
    cover = np.zeros(len(names)); n = min(safe_resources.workers(1 * 2**30), len(jobs))
    with rasterio.open(f, "w", **p) as dst, ProcessPoolExecutor(n) as ex:
        for i, nm in enumerate(names, 1):
            dst.set_band_description(i, nm)
        for r0, a in ex.map(block, jobs):
            dst.write(a, window=Window(0, r0, W, a.shape[1])); cover += (a > -32768).sum(axis=(1, 2))
    print(site.name, [(nm, round(float(c / (H * W)), 2)) for nm, c in zip(names, cover)], flush=True)


if __name__ == "__main__":
    year = int(sys.argv[1])
    for s in sys.argv[2:]:
        one(SP.site_dir(s), year)

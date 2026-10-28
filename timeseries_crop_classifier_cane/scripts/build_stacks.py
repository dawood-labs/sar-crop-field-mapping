"""Turn the per-date files into one time stack per variable, so a pixel's whole time series is one small read.

data/stacks/<name>.tif   one band per date (band description = YYYYMMDD), same grid as data/grid/index.tif,
                         tiled 256 x 256 + deflate.
data/stacks/stacks.csv   name, sensor, track, variable, n_dates, first, last, dtype, nodata
Names: S1_RO164_DES_VV, S1_RO164_DES_VH, S1_RO164_DES_angle, S1_RO099_ASC_*,
       S2_B2 .. S2_B8, S2_cs, S2_cs_cdf, S2_MSK_CLDPRB, S2_SCL, S2_QA60, S2_MSK_CLASSI_OPAQUE, S2_MSK_CLASSI_CIRRUS.
Re-run after more dates are downloaded; it rebuilds the stacks from all files present.
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import glob, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import site_paths as SP

PROJ = SP.site_dir(os.environ["SC_ROOT"]) if os.environ.get("SC_ROOT") else Path(__file__).resolve().parents[1]   # SC_ROOT = another site (external test)
OUT = PROJ / "data/stacks"


def groups():
    """(name, [(date, file, band_index)], dtype, nodata) for every variable."""
    g = {}
    for track_dir in sorted((PROJ / "data/raw/s1").glob("RO*")):
        files = sorted(track_dir.glob("S1_*.tif"))
        if not files:
            continue
        with rasterio.open(files[0]) as r:
            s1vars = [d for d in r.descriptions]                 # VV, VH (no angle in this project)
        for bi, var in enumerate(s1vars, 1):
            g[f"S1_{track_dir.name}_{var}"] = ("s1", track_dir.name, var,
                                               [(re.findall(r"_(\d{8})\.tif", f.name)[0], f, bi) for f in files])
    for sub, pat in [("s2", "S2_*.tif"), ("s2_cloud", "S2CLOUD_*.tif"), ("s2_extra", "S2X_*.tif")]:
        files = sorted((PROJ / "data/raw" / sub).glob(pat))
        if not files:
            continue
        with rasterio.open(files[0]) as r:
            names = r.descriptions
        for bi, var in enumerate(names, 1):
            g[f"S2_{var}"] = ("s2", "orbit18", var, [(re.findall(r"_(\d{8})\.tif", f.name)[0], f, bi) for f in files])
    return g


def build(name, item):
    sensor, track, var, items = item
    old = OUT / f"{name}.tif"
    if old.exists():                                       # skip stacks that already hold every date
        with rasterio.open(old) as r:
            if list(r.descriptions) == [d for d, _, _ in items]:
                return {"name": name, "sensor": sensor, "track": track, "variable": var, "n_dates": len(items),
                        "first": items[0][0], "last": items[-1][0], "dtype": r.dtypes[0], "nodata": r.nodata}
    with rasterio.open(items[0][1]) as r:
        prof = r.profile.copy()
        dtype, nodata = r.dtypes[0], r.nodata
    prof.update(count=len(items), dtype=dtype, nodata=nodata, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256, predictor=2 if np.dtype(dtype).kind in "ui" else 3,
                interleave="pixel", BIGTIFF="IF_SAFER")
    # Read every date into memory first and write the stack in ONE pass: writing band by band into
    # pixel-interleaved compressed tiles makes GDAL re-append each tile per band (huge, slow files).
    arr = np.empty((len(items), prof["height"], prof["width"]), dtype=dtype)
    for k, (d, f, bi) in enumerate(items):
        with rasterio.open(f) as src:
            arr[k] = src.read(bi)
    tmp = OUT / f"{name}.tmp"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        for k, (d, f, bi) in enumerate(items, 1):
            dst.set_band_description(k, d)
        dst.update_tags(variable=var, sensor=sensor, track=track)
    tmp.replace(OUT / f"{name}.tif")
    return {"name": name, "sensor": sensor, "track": track, "variable": var, "n_dates": len(items),
            "first": items[0][0], "last": items[-1][0], "dtype": dtype, "nodata": nodata}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = groups()
    # each worker holds one full stack in memory; size the pool from the free RAM (container-aware)
    with rasterio.open(next(iter(g.values()))[3][0][1]) as r:
        px = r.width * r.height
    def done(name, items):                                   # stack already holds every date -> no memory needed
        f = OUT / f"{name}.tif"
        if not f.exists():
            return False
        with rasterio.open(f) as r:
            return list(r.descriptions) == [d for d, _, _ in items]
    def need(v):                                             # bytes: array in its own dtype + write buffers
        with rasterio.open(v[3][0][1]) as r:
            item = np.dtype(r.dtypes[0]).itemsize
        return len(v[3]) * px * item * 2
    todo = {k: v for k, v in g.items() if not done(k, v[3])}
    biggest = max((need(v) for v in todo.values()), default=1)   # size workers by the stacks still to build
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    import safe_resources as R                               # size by the pod's memory REQUEST, not the free RAM
    workers = max(1, min(R.cpus(), len(g), int(R.safe_memory_bytes() // biggest)))
    print(f"{len(g)} stacks, {workers} workers ({R.describe()})", flush=True)
    with ThreadPoolExecutor(workers) as pool:                  # GDAL releases the GIL while (de)compressing
        rows = list(pool.map(lambda kv: build(*kv), g.items()))
    pd.DataFrame(rows).sort_values("name").to_csv(OUT / "stacks.csv", index=False)
    for r in rows:
        print(f"{r['name']:28s} {r['n_dates']:4d} dates  {r['first']}..{r['last']}")


if __name__ == "__main__":
    main()

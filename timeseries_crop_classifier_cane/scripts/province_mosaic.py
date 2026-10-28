"""Merge one district raster of every district of a province into a province raster, values as they are (no resampling:
all districts share the 10 m EPSG:32648 grid). Overlap: a valid value beats nodata, cane (1) beats 0. Clipped to the official province polygon.
usage: python scripts/province_mosaic.py <Province> <rel path in the district folder> <out name>
  e.g. python scripts/province_mosaic.py province7 data/model/s2on_r2_oct_rn5/cane_map_2026.tif province7_cane_map_2026.tif
"""
import os, shutil, sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

import site_paths as SP

prov, rel, name = sys.argv[1:4]
folder = SP.SITES / "2026" / prov
if os.environ.get("SC_SITES_PROVINCE_DIR"):           # pipeline runner: a work folder holding exactly the province's units
    folder = Path(os.environ["SC_SITES_PROVINCE_DIR"]); sites = sorted(d for d in folder.iterdir() if d.is_dir())
else:
    sites = [p for p in SP.all_sites("*_2026") if p.parent == folder]
files = [p / rel for p in sites if (p / rel).exists()]
srcs = [rasterio.open(f) for f in files]
assert len({s.crs for s in srcs}) == 1 and len({s.res for s in srcs}) == 1
nd = srcs[0].nodata
l = min(s.bounds.left for s in srcs); t = max(s.bounds.top for s in srcs)
r = max(s.bounds.right for s in srcs); b = min(s.bounds.bottom for s in srcs); res = srcs[0].res[0]
W, H = round((r - l) / res), round((t - b) / res)
out = np.full((H, W), nd, srcs[0].dtypes[0]); conflict = 0
for s in srcs:
    a = s.read(1); r0, c0 = round((t - s.bounds.top) / res), round((s.bounds.left - l) / res)
    w = out[r0:r0 + s.height, c0:c0 + s.width]
    both = (w != nd) & (a != nd); conflict += int((both & (w != a)).sum())
    take = (a != nd) & ((w == nd) | (a == 1)); w[take] = a[take]
prof = dict(srcs[0].profile, width=W, height=H, transform=Affine(res, 0, l, 0, -res, t), compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
out[~SP.province_mask(prov, out.shape, prof["transform"], prof["crs"])] = nd           # clip to the official province polygon
dst = folder / name
with rasterio.open(dst, "w", **prof) as d: d.write(out, 1)
q = files[0].with_suffix(".qml")
if q.exists(): shutil.copy(q, dst.with_suffix(".qml"))
print(dst, out.shape, f"{len(files)} districts, overlap conflicts {conflict} px",
      {int(v): round(int(c) * res * res / 4046.8564224, 3) for v, c in zip(*np.unique(out, return_counts=True)) if v != nd}, "acres")

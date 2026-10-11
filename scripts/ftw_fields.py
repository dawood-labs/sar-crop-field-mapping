"""Field boundaries from Fields of The World (FTW, ftw-tools) on our own S2 stacks: data/fields/.

Used ONLY as a reference for where fields are (e.g. one audit point per field, away from field edges). It is not a
model input: the rice map stays SAR-only (user, 2026-10-09).
FTW input = 8 bands, window A then window B, each B04 B03 B02 B08 (uint16 L2A reflectance, processing version 3 =
no +1000 offset; our stacks are S2_SR_HARMONIZED, so they match). Windows = cloud-free median per pixel:
  A = 1 Feb - 8 Mar 2026 (bare dry season, before planting)   B = 1 Oct - 15 Nov 2025 (crop canopy, before harvest)
Model: THREE_CLASS_FULL (background / field / boundary). Outputs:
  ftw_input.tif, ftw_pred.tif (1 field, 2 boundary), ftw_fields.gpkg (polygons, clipped to the AOI),
  ftw_field_id.tif (field polygon id per 10 m pixel on our grid, 0 = no field)
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401

OUT, ST = PROJ / "data/fields", PROJ / "data/stacks"
WIN = {"A": ("2026-02-01", "2026-03-08"), "B": ("2025-10-01", "2025-11-15")}
CKPT = OUT / "ftw_3class_full.ckpt"


def composite(start, end):
    with rasterio.open(ST / "S2_CLEAR.tif") as r:
        d = pd.to_datetime(list(r.descriptions))
        k = [i + 1 for i, x in enumerate(d) if pd.Timestamp(start) <= x <= pd.Timestamp(end)]
        clear = r.read(k) == 1
    out = []
    for b in ("B4", "B3", "B2", "B8"):
        with rasterio.open(ST / f"S2_{b}.tif") as r:
            a = r.read(k).astype("float32")
        a[~clear | (a == 0)] = np.nan
        with np.errstate(all="ignore"):
            out.append(np.nan_to_num(np.nanmedian(a, 0)).clip(0, 65535).astype("uint16"))
    print(f"window {start}..{end}: {len(k)} dates, no clear pixel {(~clear.any(0)).mean():.2%}", flush=True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inp = OUT / "ftw_input.tif"
    if not inp.exists():
        bands = composite(*WIN["A"]) + composite(*WIN["B"])
        with rasterio.open(ST / "S2_B4.tif") as r:
            prof = dict(r.profile, count=8, dtype="uint16", nodata=0, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(inp, "w", **prof) as dst:
            dst.write(np.stack(bands))
    if not CKPT.exists():
        subprocess.run(["ftw", "model", "download", "--type", "THREE_CLASS_FULL", "-o", str(CKPT)], check=True)
    pred = OUT / "ftw_pred.tif"
    if not pred.exists():
        # in-process with num_workers=0: the pod's /dev/shm is too small for ftw's 6 DataLoader worker processes
        import torch
        import ftw_cli.inference as fi
        torch.set_num_threads(os.cpu_count())
        DL = fi.DataLoader
        fi.DataLoader = lambda *a, **k: DL(*a, **dict(k, num_workers=0))
        fi.run(str(inp), str(CKPT), str(pred), resize_factor=2, gpu=None, patch_size=None, batch_size=2, padding=None,
               overwrite=True, mps_mode=False)
    poly = OUT / "ftw_fields_raw.gpkg"
    if not poly.exists():
        subprocess.run(["ftw", "inference", "polygonize", str(pred), "-o", str(poly), "--simplify", "0", "--min_size", "500"], check=True)
    import geopandas as gpd
    from rasterio import features
    f = gpd.read_file(poly)
    aoi = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(f.crs)
    f = f[f.intersects(aoi.union_all())].reset_index(drop=True)
    f["field_id"] = np.arange(1, len(f) + 1); f["area_ac"] = f.area / 4046.856
    f[["field_id", "area_ac", "geometry"]].to_file(OUT / "ftw_fields.gpkg", driver="GPKG")
    with rasterio.open(PROJ / "data/grid/index.tif") as g:
        prof = dict(g.profile, count=1, dtype="int32", nodata=0, compress="deflate")
        fid = features.rasterize(zip(f.geometry, f.field_id), out_shape=g.shape, transform=g.transform, fill=0, dtype="int32")
    with rasterio.open(OUT / "ftw_field_id.tif", "w", **prof) as dst:
        dst.write(fid, 1)
    print(f"fields in AOI: {len(f)}; area median {f.area_ac.median():.2f} ac, total {f.area_ac.sum():,.0f} ac; "
          f"pixels in a field {(fid > 0).mean():.1%}", flush=True)


if __name__ == "__main__":
    main()

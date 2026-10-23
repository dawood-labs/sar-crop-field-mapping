"""Label rasters from the 2025 cane polygons on the 10 m grid (first step of the label QC).

Inputs : data/source/cane2025.parquet (2025 sugarcane polygons, map dated 2025-11-29), data/grid/index.tif
Outputs: data/labels/cane2025_labels.tif, bands:
           1 label     1 cane (pixel centre inside a cane polygon), 0 non-cane, 255 outside the AOI
           2 poly_id   id of the cane polygon (0 = none); ids match data/labels/cane2025_polys.parquet
           3 edge_px   cane: distance in pixels to the nearest non-cane pixel; non-cane: distance to the nearest cane
                       pixel (both capped at 50) -> used to drop edge pixels on both sides
         data/labels/cane2025_polys.parquet  the cane polygons inside the AOI (+ poly_id, area_acres, n_px)
usage: python scripts/labels_prep.py
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import json, time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage
import site_paths as SP

PROJ = SP.site_dir(os.environ["SC_ROOT"]) if os.environ.get("SC_ROOT") else Path(__file__).resolve().parents[1]   # SC_ROOT = another site (external test)
ACRE_M2 = 4046.856


def main():
    t0 = time.time()
    out = PROJ / "data/labels"; out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(PROJ / "data/grid/index.tif") as g:
        prof, tr, shape = g.profile.copy(), g.transform, g.shape
        inside = g.read(2).astype(bool)
    aoi = gpd.read_file(os.environ.get("SC_AOI") or PROJ / "data/aoi/aoi_site37.shp")
    c = gpd.read_parquet(Path(__file__).resolve().parents[1] / "data/source/cane2025.parquet").to_crs(prof["crs"])
    c = c[c.intersects(aoi.to_crs(prof["crs"]).union_all().buffer(100))].reset_index(drop=True)
    c["poly_id"] = np.arange(1, len(c) + 1, dtype="int32")
    pid = features.rasterize(zip(c.geometry, c.poly_id), out_shape=shape, transform=tr, fill=0, dtype="int32")
    cane = pid > 0
    d_in = ndimage.distance_transform_edt(cane)            # cane px -> distance to nearest non-cane px
    d_out = ndimage.distance_transform_edt(~cane)          # non-cane px -> distance to nearest cane px
    edge = np.where(cane, d_in, d_out).clip(0, 50).astype("uint8")
    lab = np.where(inside, cane.astype("uint8"), 255).astype("uint8")
    pid = np.where(inside, pid, 0)
    c["n_px"] = np.bincount(pid[pid > 0], minlength=len(c) + 1)[1:]
    c["area_acres"] = c.area / ACRE_M2
    c.to_parquet(out / "cane2025_polys.parquet")
    p = dict(prof, count=3, dtype="int32", nodata=None, compress="deflate")
    with rasterio.open(out / "cane2025_labels.tif", "w", **p) as dst:
        dst.write(np.stack([lab.astype("int32"), pid, edge.astype("int32")]))
        for i, n in enumerate(["label", "poly_id", "edge_px"], 1):
            dst.set_band_description(i, n)
    s = {"cane_polygons": int(len(c)), "polygons_with_px_in_aoi": int((c.n_px > 0).sum()),
         "cane_px": int((lab == 1).sum()), "noncane_px": int((lab == 0).sum()),
         "cane_acres_px": float((lab == 1).sum() * 100 / ACRE_M2), "aoi_acres_px": float(inside.sum() * 100 / ACRE_M2),
         "cane_interior_px_edge>=2": int(((lab == 1) & (edge >= 2)).sum()),
         "noncane_px_edge>=2": int(((lab == 0) & (edge >= 2)).sum()),
         "polygon_size_acres": c.area_acres.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_dict()}
    (out / "cane2025_labels_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(s, indent=1, default=float)); print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

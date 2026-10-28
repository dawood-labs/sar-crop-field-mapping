"""Summary of the field-delineation review versions of one province (27 Oct 2026): per version polygons, acres, whole / mixed_whole / cut,
static-raster sugarcane inside / left out, cassava and empty land inside -> fd_review/summary.csv + README.txt.
usage: python scripts/fd_review_summary.py <Province>"""
import glob, os, sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

prov = sys.argv[1]; p = prov.lower(); O = f"sites/2026/{prov}/fd_review"
B = gpd.read_file(glob.glob(f"data/delineation/{p}_2026/*Boundary.gpkg")[0]).to_crs(32648)
with rasterio.open(f"sites/2026/{prov}/{p}_class_2026_s20.tif") as r: a = r.read(1); tr = r.transform
inb = rasterize([(g, 1) for g in B.geometry], out_shape=a.shape, transform=tr, dtype="uint8") == 1
A = lambda m: round(float(m.sum()) * 100 / 4046.8564224); rows = []
for f in sorted(glob.glob(f"{O}/*_fd_sugarcane_2026_*.gpkg")):
    c = gpd.read_file(f); m = rasterize([(g, 1) for g in c.geometry], out_shape=a.shape, transform=tr, dtype="uint8") == 1
    rv = c.review.value_counts()
    rows.append(dict(version=os.path.basename(f).split("_2026_")[1][:-5], polygons=len(c), area_ac=round(c.area_ac.sum()),
                     whole=int(rv.get("whole", 0)), mixed_whole=int(rv.get("mixed_whole", 0)), cut=int(rv.get("cut", 0)),
                     raster_sugarcane_in_boundary_ac=A((a == 1) & inb), sugarcane_inside_ac=A((a == 1) & m), sugarcane_left_out_ac=A((a == 1) & inb & ~m),
                     cassava_inside_ac=A((a == 281) & m), empty_inside_ac=A((a == 0) & m)))
s = pd.DataFrame(rows); s.to_csv(f"{O}/summary.csv", index=False); print(s.to_string(index=False))
open(f"{O}/README.txt", "w").write(f"""{prov} - sugarcane field polygons from the field delineation, 4 versions for review (26-27 Oct 2026).
All: delineation outlines (no simplify), sugarcane share from the static colour model raster (pixel centres, cloud not counted),
cut at the delineation boundary, overlaps / tails / small gaps cleaned, 0.2 m between neighbours, area_ac in EPSG:32648, 3 decimals.
Columns: field_id, predicted (1 = sugarcane), area_ac, cane_pct, cloud_pct, review, uid.
review: whole = field kept as delineated (>= 75 % cane); mixed_whole = kept whole although < 75 % cane (non-cane part small);
        cut = piece cut out of a big mixed field with straight lines.
Versions:
  v1    every field below 75 % cane is cut (first rule)
  1ac   a field below 75 % cane is kept whole while its non-cane part (empty + cassava) <= 1 acre, else cut
  1p5ac same with 1.5 acres
  2ac   same with 2 acres
  (fields with < 20 % AND < 0.2 ac sugarcane are dropped in the threshold versions)
summary.csv: polygons, acres, whole / mixed_whole / cut counts, and how much static-raster sugarcane is inside / left out and how much
cassava / empty land is inside each version.
""")

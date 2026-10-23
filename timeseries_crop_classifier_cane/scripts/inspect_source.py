"""Look at the shared source data once and cache it as GeoParquet (standing-cane test AOI selection).

Inputs : data/source/provinces/*.shp   47 province polygons (ADM1_EN, ADM1_TH), EPSG:4326
         data/source/cane2025/*.shp    2025 sugarcane field map (359,680 polygons, column predicted), EPSG:4326
Outputs: data/source/provinces.parquet
         data/source/cane2025.parquet      + area_acres (equal-area EPSG:6933) and the province of each centroid
         data/source/cane_by_province.csv  cane acres, share of the province, polygon count, field size
The 930 MB shapefile is read in chunks so memory stays well under the pod request.
usage: python scripts/inspect_source.py
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

S = Path(__file__).resolve().parents[1] / "data/source"
CANE = next((S / "cane2025").glob("*.shp"))
PROV = next((S / "provinces").glob("*.shp"))
ACRE_M2 = 4046.856
EA = "EPSG:6933"
CHUNK = 60_000


def main():
    t0 = time.time()
    p = gpd.read_file(PROV)
    p["area_acres"] = p.to_crs(EA).area / ACRE_M2
    p.to_parquet(S / "provinces.parquet")
    n = pyogrio.read_info(CANE)["features"]
    parts = []
    for s in range(0, n, CHUNK):
        c = gpd.read_file(CANE, skip_features=s, max_features=CHUNK, engine="pyogrio")
        ce = c.to_crs(EA)
        c["area_acres"] = ce.area / ACRE_M2
        cen = gpd.GeoDataFrame(geometry=c.geometry.representative_point(), crs=c.crs)
        j = gpd.sjoin(cen, p[["ADM1_EN", "geometry"]], how="left", predicate="within")
        c["province"] = j.groupby(level=0).ADM1_EN.first().reindex(c.index).values
        parts.append(c)
        print(f"  {s + len(c):,}/{n:,} ({time.time() - t0:.0f} s)", flush=True)
    c = pd.concat(parts, ignore_index=True)
    c = gpd.GeoDataFrame(c, crs=parts[0].crs)
    c.to_parquet(S / "cane2025.parquet", compression="zstd", write_covering_bbox=True)
    print("predicted values:", c.predicted.value_counts().to_dict())
    t = c.groupby("province").agg(cane_acres=("area_acres", "sum"), polygons=("area_acres", "size"),
                                  median_field_ac=("area_acres", "median"))
    t = t.join(p.set_index("ADM1_EN")[["ADM1_TH", "area_acres"]].rename(columns={"area_acres": "province_acres"}))
    t["cane_share_pct"] = 100 * t.cane_acres / t.province_acres
    t = t.sort_values("cane_acres", ascending=False)
    t.round(2).to_csv(S / "cane_by_province.csv")
    print(f"cane total {c.area_acres.sum():,.0f} ac, {len(c):,} polygons; outside the 47 provinces: {int(c.province.isna().sum()):,}")
    print(t.round(2).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

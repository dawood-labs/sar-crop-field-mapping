"""Rank districts for a standing-cane test AOI: lots of cane, normal size (user / manager request, 11 Oct 2026).

The shared AOI file only has provinces (0.2 - 5 million acres each), far too big for a test, so districts come from
FAO GAUL 2015 level 2 on Earth Engine (country filter only, geometry as published).
Inputs : data/source/cane2025.parquet (from inspect_source.py), data/source/provinces.parquet, Earth Engine
Outputs: data/source/districts.parquet         district polygons inside the 47 provinces
         data/source/cane_by_district.csv       district acres, 2025 cane acres, cane share, polygons, median field
usage: python scripts/pick_aoi.py
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import sys, time
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "data/source"
ACRE_M2 = 4046.856
EA = "EPSG:6933"
EE_ROOT = os.environ.get("EE_HELPERS_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.join(EE_ROOT, "scripts"))


def districts():
    out = S / "districts.parquet"
    if out.exists():
        return gpd.read_parquet(out)
    import ee
    from sar_pipeline.auth import init_ee
    init_ee({"_root": EE_ROOT, "auth": {"key_file": os.environ.get("EE_KEY_FILE", __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json")),
                                        "project": os.environ.get("EE_PROJECT", __import__("os").environ.get("EE_PROJECT", ""))}})
    fc = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq("ADM0_NAME", "the country")) \
        .select(["ADM1_NAME", "ADM2_NAME", "ADM2_CODE"])
    d = ee.data.computeFeatures({"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    d = d.set_crs(4326)
    p = gpd.read_parquet(S / "provinces.parquet")
    cen = gpd.GeoDataFrame(geometry=d.geometry.representative_point(), crs=4326)
    j = gpd.sjoin(cen, p[["ADM1_EN", "geometry"]], how="left", predicate="within")
    d["province"] = j.groupby(level=0).ADM1_EN.first().reindex(d.index).values
    d = d[d.province.notna()].reset_index(drop=True)
    d["district_acres"] = d.to_crs(EA).area / ACRE_M2
    d.to_parquet(out)
    return d


def main():
    t0 = time.time()
    d = districts()
    print(f"{len(d)} districts in the 47 provinces ({time.time() - t0:.0f} s)", flush=True)
    c = gpd.read_parquet(S / "cane2025.parquet", columns=["area_acres", "geometry"])
    cen = gpd.GeoDataFrame({"area_acres": c.area_acres.values}, geometry=c.geometry.representative_point(), crs=c.crs)
    j = gpd.sjoin(cen, d[["ADM2_CODE", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    t = j.groupby("ADM2_CODE").agg(cane_acres=("area_acres", "sum"), polygons=("area_acres", "size"), median_field_ac=("area_acres", "median"))
    t = d.drop(columns="geometry").set_index("ADM2_CODE").join(t).fillna({"cane_acres": 0, "polygons": 0})
    t["cane_share_pct"] = 100 * t.cane_acres / t.district_acres
    t = t.sort_values("cane_acres", ascending=False)
    t.round(2).to_csv(S / "cane_by_district.csv")
    print(f"cane assigned to a district {j.ADM2_CODE.notna().mean():.1%}; district size median {t.district_acres.median():,.0f} ac")
    sel = t[(t.district_acres.between(80_000, 300_000))].sort_values("cane_share_pct", ascending=False)
    print("Top districts by cane share, size 80k - 300k acres:")
    print(sel.head(15)[["province", "ADM2_NAME", "district_acres", "cane_acres", "cane_share_pct", "polygons", "median_field_ac"]]
          .round(1).to_string())
    print("Top districts by cane acres (any size):")
    print(t.head(10)[["province", "ADM2_NAME", "district_acres", "cane_acres", "cane_share_pct"]].round(1).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

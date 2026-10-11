"""Clean the field delineation once, so every output built from it is small and topologically clean.

Source: data/fields/delineation/fields_raw.gpkg (untouched S3 download, kept).
Output: data/fields/delineation/fields_clean.parquet (GeoParquet; read by field_map_v5.py), all attributes, plus
        n_merged / merged_uids (nested polygons merged into this one). Steps (scripts/topology.py): simplify 0.5 m ->
        merge nested (>= 90 % inside a larger polygon) -> separate (no overlap, no touching, 0.1 m gap).
Also rebuilds data/fields/delineation_field_id.tif + delineation_fields_attrs.csv (field id per 10 m pixel, AOI polygons).
Re-running always starts from the raw file.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import sys, time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
import topology as tp  # noqa: E402

D = PROJ / "data/fields/delineation"
RAW, OUT = D / "fields_raw.gpkg", D / "fields_clean.parquet"


def main():
    import geopandas as gpd, shapely, rasterio
    from rasterio import features
    t0 = time.time()
    g = gpd.read_file(RAW, layer="fields", engine="pyogrio")
    n0, v0, u0 = len(g), int(shapely.get_num_coordinates(g.geometry.values).sum()), None
    print(f"raw {n0:,} polygons, {v0:,} vertices ({time.time() - t0:.0f} s)", flush=True)
    G = tp.simplify(g.geometry.values)
    G, keep, host = tp.merge_nested(G)
    uid = g.uid.values
    g["n_merged"] = np.bincount(host[host >= 0], minlength=len(g)) if (host >= 0).any() else 0
    mu = {}
    for s, b in zip(np.nonzero(host >= 0)[0], host[host >= 0]):
        mu.setdefault(b, []).append(uid[s])
    g["merged_uids"] = [",".join(mu.get(i, [])) or None for i in range(len(g))]
    g["geometry"] = G
    g = g[keep].reset_index(drop=True)
    print(f"simplified + {int((~keep).sum()):,} nested polygons merged ({time.time() - t0:.0f} s)", flush=True)
    a_before = g.area.sum()
    g["geometry"] = tp.separate(g.geometry.values)
    g = g[~g.is_empty].reset_index(drop=True)
    g["area_m2"] = g.area; g["area_acres"] = g.area / 4046.856
    n_int, gap = tp.check(g.geometry.values)
    print(f"separated: {len(g):,} polygons, area change {(g.area.sum() - a_before) / a_before:+.3%}, intersecting pairs {n_int}, "
          f"min gap {gap:.3f} m, vertices {int(shapely.get_num_coordinates(g.geometry.values).sum()):,} ({time.time() - t0:.0f} s)", flush=True)
    tmp = OUT.with_name(OUT.stem + ".tmp.parquet")
    g.to_parquet(tmp, compression="zstd")
    tmp.replace(OUT)
    print(f"{OUT.name}: {OUT.stat().st_size / 1e6:,.0f} MB (raw {RAW.stat().st_size / 1e6:,.0f} MB)", flush=True)

    aoi = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(g.crs).union_all()
    f = g[g.intersects(aoi)].reset_index(drop=True); f["field_id"] = np.arange(1, len(f) + 1)
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        prof = dict(r.profile, count=1, dtype="int32", nodata=0, compress="deflate")
        fid = features.rasterize(zip(f.geometry, f.field_id), out_shape=r.shape, transform=r.transform, fill=0, dtype="int32")
    with rasterio.open(PROJ / "data/fields/delineation_field_id.tif", "w", **prof) as dst:
        dst.write(fid, 1)
    f[["field_id", "uid", "area_acres", "Confidence", "Shp_Label", "n_merged"]].to_csv(PROJ / "data/fields/delineation_fields_attrs.csv", index=False)
    print(f"delineation_field_id.tif: {len(f):,} fields in AOI; done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

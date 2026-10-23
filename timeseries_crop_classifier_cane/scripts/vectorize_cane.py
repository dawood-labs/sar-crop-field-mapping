"""Sugarcane polygons from the cleaned layer-2 map, merged per province (user, 22 Oct 2026).
Per district: <site>/data/model/layer2_final_sieve/class_2026.tif (made by sieve_layer2.py apply, SIZE=20) -> cane pixels (1)
vectorized (4-connected, pixel edges, no smoothing) in the raster's metric CRS -> area_ac (acres, 3 decimals) -> polygons
< MIN_AC (0.15 ac) dropped -> <site>/data/model/layer2_final_sieve/cane_polygons_2026.gpkg.
Province: all districts merged into one GeoPackage + QGIS style (company Sugarcane #33ef2a).
usage: python scripts/vectorize_cane.py <province folder name, e.g. province7> [out.gpkg]
       env MIN_AC=0.15, SRC=layer2_final_sieve
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

import safe_resources as R
import site_paths as SP

ACRE = 4046.8564224
MIN_AC = float(os.environ.get("MIN_AC", 0.15))
SRC = os.environ.get("SRC", "layer2_final_sieve")
QML = """<!DOCTYPE qgis><qgis version="3.34" styleCategories="Symbology"><renderer-v2 type="singleSymbol"><symbols>
<symbol type="fill" name="0" alpha="1"><layer class="SimpleFill"><Option type="Map">
<Option name="color" value="51,239,42,255" type="QString"/><Option name="outline_color" value="20,120,15,255" type="QString"/>
<Option name="outline_width" value="0.1" type="QString"/><Option name="style" value="solid" type="QString"/>
</Option></layer></symbol></symbols></renderer-v2></qgis>"""


def one(site):
    d = SP.site_dir(site) / "data/model" / SRC
    with rasterio.open(d / "class_2026.tif") as r: a, tr, crs = r.read(1), r.transform, r.crs
    m = a == 1
    geoms = [shape(g) for g, _ in shapes(m.astype(np.uint8), mask=m, transform=tr, connectivity=4)]
    g = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    g["area_ac"] = (g.area / ACRE).round(3)
    n0, ac0 = len(g), g.area_ac.sum()
    g = g[g.area_ac >= MIN_AC].reset_index(drop=True)
    name = site.replace("_2026", "")
    g.insert(0, "predicted", np.int32(1)); g.insert(1, "class_name", "Sugarcane"); g["district"] = name
    g.to_file(d / "cane_polygons_2026.gpkg", layer="sugarcane", driver="GPKG")
    (d / "cane_polygons_2026.qml").write_text(QML)
    return dict(site=site, n_all=n0, n_kept=len(g), ac_all=round(ac0, 3), ac_kept=round(g.area_ac.sum(), 3),
                ac_dropped_small=round(ac0 - g.area_ac.sum(), 3), crs=str(crs))


if __name__ == "__main__":
    prov = sys.argv[1]; t0 = time.time()
    sites = [p.name for p in SP.all_sites("*_2026") if p.parent.name == prov and (p / "data/model" / SRC / "class_2026.tif").exists()]
    with ProcessPoolExecutor(min(len(sites), R.cpus())) as ex: rows = list(ex.map(one, sites))
    for r in rows: print(r)
    crs = {r["crs"] for r in rows}; assert len(crs) == 1, f"districts in different CRS: {crs}"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else SP.SITES / "2026" / prov / f"{prov.lower()}_sugarcane_2026.gpkg"
    g = pd.concat([gpd.read_file(SP.site_dir(s) / "data/model" / SRC / "cane_polygons_2026.gpkg") for s in sites], ignore_index=True)
    g["province"] = prov.replace("_", " "); g["year"] = 2026
    g = gpd.GeoDataFrame(g[["predicted", "class_name", "area_ac", "district", "province", "year", "geometry"]], crs=crs.pop())
    g.insert(0, "fid_cane", np.arange(1, len(g) + 1, dtype=np.int64))
    if out.exists(): out.unlink()
    g.to_file(out, layer="sugarcane", driver="GPKG"); out.with_suffix(".qml").write_text(QML)
    s = pd.DataFrame(rows).drop(columns="crs"); s.loc[len(s)] = ["TOTAL", s.n_all.sum(), s.n_kept.sum(), s.ac_all.sum(), s.ac_kept.sum(), s.ac_dropped_small.sum()]
    s.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    (out.with_name(out.stem + "_meta.json")).write_text(json.dumps(dict(
        source=f"<site>/data/model/{SRC}/class_2026.tif (layer 1 s2on_r2_oct_rn5 + layer 2 colour v8 = layer2_final, gaps filled / specks "
               f"removed < 20 px)", min_ac=MIN_AC, crs=str(g.crs), polygons=len(g), area_ac=round(float(g.area_ac.sum()), 3), districts=sites), indent=1))
    print(s.to_string(index=False)); print(out, len(g), "polygons", f"{time.time() - t0:.0f} s")

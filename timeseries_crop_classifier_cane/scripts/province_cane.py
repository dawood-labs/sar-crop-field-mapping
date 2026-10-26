"""Province-level sugarcane polygons with NO cut at district lines (user, 22 Oct 2026).
1. mosaic every district's <site>/data/model/layer2_final/class_2026.tif of the province (all on the same 10 m EPSG:32648 grid;
   where two district AOIs overlap and disagree, sugarcane wins over cassava over none)
2. clean the WHOLE province map with sieve_layer2.run (fill non-cane holes < SIZE px enclosed by cane only, drop isolated cane
   specks < SIZE px) -> sites/2026/<Province>/<prov>_class_2026_s<SIZE>.tif (+ .qml)
2b. pixels outside the official province polygon (data/source/provinces.parquet) are set to nodata (clip)
3. cane px (1) -> polygons (4-connected pixel edges, no smoothing), area_ac = polygon area in EPSG:32648 (UTM 48N, metres)
   / 4046.8564224, rounded to 3 decimals; polygons < MIN_AC dropped; one field only: area_ac
   -> sites/2026/<Province>/<prov>_sugarcane_2026.gpkg (layer sugarcane, + .qml, _meta.json)
usage: python scripts/province_cane.py province7        env SIZE=20 MIN_AC=0.15
"""
import json, os, sys, time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape

import site_paths as SP
from sieve_layer2 import ND, run
from vectorize_cane import ACRE, QML

SIZE, MIN_AC = int(os.environ.get("SIZE", 20)), float(os.environ.get("MIN_AC", 0.15))
RANK = {ND: 0, 0: 1, 281: 2, 1: 3}                      # overlap priority


def mosaic(sites):
    srcs = [rasterio.open(s / "data/model/layer2_final/class_2026.tif") for s in sites]
    crs = {s.crs for s in srcs}; assert len(crs) == 1, crs
    l = min(s.bounds.left for s in srcs); t = max(s.bounds.top for s in srcs)
    r = max(s.bounds.right for s in srcs); b = min(s.bounds.bottom for s in srcs)
    W, H = round((r - l) / 10), round((t - b) / 10); tr = Affine(10, 0, l, 0, -10, t)
    out = np.full((H, W), ND, np.uint16); rk = np.zeros((H, W), np.uint8); conflict = 0
    lut = np.zeros(65536, np.uint8)
    for k, v in RANK.items(): lut[k] = v
    for s in srcs:
        a = s.read(1); r0, c0 = round((t - s.bounds.top) / 10), round((s.bounds.left - l) / 10)
        win = out[r0:r0 + s.height, c0:c0 + s.width]; wr = rk[r0:r0 + s.height, c0:c0 + s.width]; ar = lut[a]
        conflict += int(((wr > 0) & (ar > 0) & (win != a)).sum())
        take = ar > wr; win[take] = a[take]; wr[take] = ar[take]
    prof = dict(srcs[0].profile, width=W, height=H, transform=tr, compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    return out, prof, conflict


def main(prov):
    t0 = time.time(); folder = SP.SITES / "2026" / prov; stem = prov.lower()
    sites = [p for p in SP.all_sites("*_2026") if p.parent == folder and (p / "data/model/layer2_final/class_2026.tif").exists()]
    a, prof, conflict = mosaic(sites); print(f"mosaic {a.shape} from {len(sites)} districts, overlap conflicts {conflict} px ({time.time() - t0:.0f} s)")
    inside = SP.province_mask(prov, a.shape, prof["transform"], prof["crs"])       # clip to the official province polygon
    clipped = int(((a != ND) & ~inside).sum()); a[~inside] = ND
    print(f"clipped to the province boundary: {clipped * 100 / ACRE:,.0f} ac of district area outside the province removed")
    b = run(a, SIZE); print(f"cleaned ({time.time() - t0:.0f} s)")
    f = folder / f"{stem}_class_2026_s{SIZE}.tif"
    with rasterio.open(f, "w", **prof) as d: d.write(b, 1)
    f.with_suffix(".qml").write_text((sites[0] / "data/model/layer2_final/class_2026.qml").read_text())
    m = b == 1
    g = gpd.GeoDataFrame(geometry=[shape(x) for x, _ in shapes(m.astype(np.uint8), mask=m, transform=prof["transform"], connectivity=4)],
                         crs=prof["crs"])
    g["area_ac"] = (g.area / ACRE).round(3); n0, ac0 = len(g), float(g.area_ac.sum())
    g = g[g.area_ac >= MIN_AC].reset_index(drop=True)[["area_ac", "geometry"]]
    out = folder / f"{stem}_sugarcane_2026.gpkg"
    if out.exists(): out.unlink()
    g.to_file(out, layer="sugarcane", driver="GPKG"); out.with_suffix(".qml").write_text(QML)
    meta = dict(province=prov, districts=[s.name for s in sites], size_px=SIZE, min_ac=MIN_AC, crs=str(g.crs),
                polygons=len(g), area_ac=round(float(g.area_ac.sum()), 3), dropped_small=dict(n=n0 - len(g), ac=round(ac0 - g.area_ac.sum(), 3)),
                overlap_conflict_px=conflict, before_clean_sugarcane_ac=round(float((a == 1).sum()) * 100 / ACRE, 3),
                after_clean_sugarcane_ac=round(float(m.sum()) * 100 / ACRE, 3))
    for x in (folder / f"{stem}_sugarcane_2026_summary.csv",): x.unlink(missing_ok=True)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=1)); print(json.dumps(meta, indent=1)); print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])

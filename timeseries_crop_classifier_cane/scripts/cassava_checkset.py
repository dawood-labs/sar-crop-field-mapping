"""QGIS check set for the static cassava model: N blocks per class from <site>/data/model/cassava_static_<vtag>/class_2026.tif
(interior 5 x 5 uniform, >= 300 m from the user's polygons), written as 30 m squares with the model class and probability and
an empty `user_call` column (sugarcane / cassava / other) for the user to fill in QGIS.
usage: python scripts/cassava_checkset.py <vtag> <site_dir> [n_per_class=20]  -> sites/user_labels/checkset_<vtag>_<site>.gpkg
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box
import site_paths as SP
P = Path(__file__).resolve().parents[1]; vtag, site = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
S = SP.site_dir(site) / "data"; key = {"site25_2026": "site25", "site41_2026": "Site41", "site33_2026": "Site33"}.get(site, site)
with rasterio.open((S / f"model/cassava_colour_{vtag}" if (S / f"model/cassava_colour_{vtag}").exists() else S / f"model/cassava_static_{vtag}") / "class_2026.tif") as r: cls, tr = r.read(1), r.transform
with rasterio.open((S / f"model/cassava_colour_{vtag}" if (S / f"model/cassava_colour_{vtag}").exists() else S / f"model/cassava_static_{vtag}") / "cassava_prob_2026.tif") as r: prob = r.read(1)
polys = pd.concat([gpd.read_file(P / "sites/user_labels/gcps_country_2026-10-15_analysis.gpkg").to_crs(32648)[["site", "geometry"]],
                   gpd.read_file(P / "sites/user_labels/cane_polygons_2026-10-15.gpkg").to_crs(32648)[["site", "geometry"]],
                   gpd.read_file(P / "sites/user_labels/cassava_noise_site25_2026-10-15.gpkg").to_crs(32648).assign(site="site25")[["site", "geometry"]]])
far = np.ones(cls.shape, bool); pp = polys[polys.site == key]
if len(pp): far &= ~rasterize([(g.buffer(300), 1) for g in pp.geometry], out_shape=cls.shape, transform=tr, dtype="uint8").astype(bool)
k5 = np.ones((5, 5), bool); cv = cls.astype("int32"); uni = ndimage.minimum_filter(cv, footprint=k5) == ndimage.maximum_filter(cv, footprint=k5)
rng = np.random.default_rng(3); rows = []
for c, name in ((281, "cassava_candidate"), (1, "sugarcane")):
    rr, cc = np.nonzero(uni & far & (cls == c)); k = rng.choice(len(rr), min(n, len(rr)), replace=False)
    for r_, c_ in zip(rr[k], cc[k]):
        x, y = tr * (c_ + 0.5, r_ + 0.5)
        rows.append(dict(model_class=name, cassava_prob=int(prob[r_, c_]), row=int(r_), col=int(c_), user_call="", geometry=box(x - 15, y - 15, x + 15, y + 15)))
g = gpd.GeoDataFrame(rows, crs=32648).sample(frac=1, random_state=3).reset_index(drop=True); g.insert(0, "id", [f"C{i + 1:02d}" for i in range(len(g))])
out = P / "sites/user_labels" / f"checkset_{vtag}_{site}.gpkg"; g.to_file(out, driver="GPKG", layer="checkset"); print(len(g), "blocks ->", out); print(g.model_class.value_counts().to_dict())

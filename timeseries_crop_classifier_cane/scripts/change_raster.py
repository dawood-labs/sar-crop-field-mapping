"""change_2025_2026.tif for a 2026 district: 2025 = the 2025 cane map (label), 2026 = the model's 2026 map.
0 non-cane both, 1 cane both, 2 cane 2025 only (lost), 3 cane 2026 only (new), 255 outside; + .qml and acres.
(The first AOI also has a model-vs-model change raster in sites/site37_2026/data/model/m2026/.)
usage: python scripts/change_raster.py sites/<site> [model tag, default m2026]
"""
import json, sys
from pathlib import Path
import numpy as np, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qgis_style as Q
import site_paths as SP

site = SP.site_dir(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "m2026"
d = site / "data/model" / tag
lab = rasterio.open(site / "data/labels/cane2025_labels.tif").read(1)
with rasterio.open(d / "cane_map_2026.tif") as r:
    m, prof = r.read(1), r.profile.copy()
c = np.full(lab.shape, 255, "uint8"); ok = (lab != 255) & (m != 255)
for v, (a, b) in enumerate(((0, 0), (1, 1), (1, 0), (0, 1))):
    c[ok & (lab == a) & (m == b)] = v
with rasterio.open(d / "change_2025_2026.tif", "w", **dict(prof, dtype="uint8", nodata=255, compress="deflate")) as dst:
    dst.write(c, 1); dst.set_band_description(1, "0 non-cane both, 1 cane both, 2 cane 2025 map only (lost), 3 cane 2026 model only (new)")
Q.paletted(d / "change_2025_2026.tif", {0: ("#e6e6e6", "non-cane both years"), 1: ("#33ef2a", "cane both years"),
                                        2: ("#e63333", "cane 2025 only (lost)"), 3: ("#1a66ff", "cane 2026 only (new)")})
ac = {k: round(float((c == v).sum() * 100 / 4046.8564224)) for k, v in (("noncane_both", 0), ("cane_both", 1), ("lost", 2), ("new", 3))}
a25, a26 = ac["cane_both"] + ac["lost"], ac["cane_both"] + ac["new"]
s = {"cane_2025_map_ac": a25, "cane_2026_model_ac": a26, "change_pct": round(100 * (a26 - a25) / max(a25, 1), 1), **ac}
(d / "change_2025_2026.json").write_text(json.dumps(s, indent=1)); print(site.name, s)

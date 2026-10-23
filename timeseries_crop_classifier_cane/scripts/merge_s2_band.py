"""Add a separately downloaded S2 band (e.g. B2 in data/raw/s2_b2/) into the per-date chip files data/raw/s2/S2_<date>.tif,
in band order B2, B3, B4, B5, B8, and write a QGIS style per file (5-3-2 composite: R = B5, G = B3, B = B2).
usage: SC_ROOT=sites/<site> python scripts/merge_s2_band.py s2_b2
"""
from __future__ import annotations

import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import site_paths as SP

ROOT = SP.site_dir(os.environ["SC_ROOT"])
ORDER = ["B2", "B3", "B4", "B5", "B8"]
QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories"><pipe>
 <rasterrenderer type="multibandcolor" redBand="4" greenBand="2" blueBand="1" opacity="1" alphaBand="-1">
  <redContrastEnhancement><minValue>500</minValue><maxValue>3000</maxValue><algorithm>StretchToMinimumMaximum</algorithm></redContrastEnhancement>
  <greenContrastEnhancement><minValue>300</minValue><maxValue>1800</maxValue><algorithm>StretchToMinimumMaximum</algorithm></greenContrastEnhancement>
  <blueContrastEnhancement><minValue>150</minValue><maxValue>1400</maxValue><algorithm>StretchToMinimumMaximum</algorithm></blueContrastEnhancement>
 </rasterrenderer></pipe></qgis>
"""


def one(f, extra_dir):
    e = ROOT / "data/raw" / extra_dir / f.name
    if not e.exists():
        return f.name, "missing extra band"
    with rasterio.open(f) as a, rasterio.open(e) as b:
        bands = {d: a.read(i + 1) for i, d in enumerate(a.descriptions)}
        bands.update({d: b.read(i + 1) for i, d in enumerate(b.descriptions)})
        prof, tags = a.profile.copy(), a.tags()
    if not all(k in bands for k in ORDER):
        return f.name, f"bands {sorted(bands)}"
    tmp = f.with_suffix(".merge.tif")
    prof.update(count=len(ORDER), compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(np.stack([bands[k] for k in ORDER]))
        for i, k in enumerate(ORDER, 1):
            dst.set_band_description(i, k)
        dst.update_tags(**tags)
    tmp.replace(f)
    f.with_suffix(".qml").write_text(QML)
    return f.name, "ok"


def main():
    extra = sys.argv[1]
    files = sorted((ROOT / "data/raw/s2").glob("S2_*.tif"))
    with ThreadPoolExecutor(os.cpu_count() or 8) as ex:
        res = list(ex.map(lambda f: one(f, extra), files))
    bad = [r for r in res if r[1] != "ok"]
    print(f"{len(res) - len(bad)} / {len(res)} merged", bad[:5])


if __name__ == "__main__":
    main()

"""Clean the layer-2 class map (user, 22 Oct 2026): fill small gaps INSIDE cane fields and remove tiny isolated cane specks,
WITHOUT moving any existing cane boundary (a plain GDAL sieve also merged small non-cane pieces at field edges into cane).
  1. gap fill : a non-cane component (none 0 / cassava 281; 8-connected, so a diagonal channel to the outside counts as open)
                with < N px that touches ONLY cane (not 65535, not the image edge) becomes sugarcane (1).
  2. de-noise : a cane component (8-connected) with < N px becomes the class of its nearest non-cane pixel.
  Nothing else changes; 65535 (outside AOI / cloud) is never changed.
Input <site>/data/model/layer2_final/class_2026.tif (locked, never overwritten).

  python scripts/sieve_layer2.py test <site> [...]           -> <site>/data/model/layer2_final_sieve_test/: class_s<N>.tif,
                                                                change_s<N>.tif, sieve_test.csv   (env SIZES=4,8,12,16,20,30)
  SIZE=12 python scripts/sieve_layer2.py apply <site> [...]   -> <site>/data/model/layer2_final_sieve/class_2026.tif, change_2026.tif, sieve.json
change raster: 1 gap filled -> sugarcane, 2 cane speck removed, 0 unchanged.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

import safe_resources as R
import site_paths as SP

ND, AC = 65535, 100 / 4046.8564224     # 10 m pixel = 100 m2
CHANGE_QML = """<!DOCTYPE qgis><qgis version="3.34"><pipe><rasterrenderer type="paletted" band="1" opacity="1"><colorPalette>
<paletteEntry value="0" color="#000000" alpha="0" label="unchanged"/>
<paletteEntry value="1" color="#1f78ff" alpha="255" label="gap filled -> sugarcane"/>
<paletteEntry value="2" color="#ff2a2a" alpha="255" label="cane speck removed"/>
</colorPalette></rasterrenderer></pipe></qgis>"""


def load(site):
    with rasterio.open(site / "data/model/layer2_final/class_2026.tif") as r: return r.read(1), r.profile.copy()


S8 = np.ones((3, 3), bool)


def run(a, size):
    valid, cane = a != ND, a == 1
    out = a.copy()
    bg = valid & ~cane                                                   # 1. holes: small non-cane parts enclosed by cane only
    lab, n = ndi.label(bg, S8)
    open_ = ndi.binary_dilation(~valid, S8); open_[0, :] = open_[-1, :] = open_[:, 0] = open_[:, -1] = True
    touch = np.zeros(n + 1, bool); touch[np.unique(lab[open_ & bg])] = True
    small = np.bincount(lab.ravel(), minlength=n + 1) < size
    hole = small & ~touch; hole[0] = False
    out[hole[lab]] = 1
    lab, n = ndi.label(cane, S8)                                         # 2. specks: small isolated cane parts
    small = np.bincount(lab.ravel(), minlength=n + 1) < size; small[0] = False
    speck = small[lab]
    if speck.any():
        _, (ri, ci) = ndi.distance_transform_edt(~bg, return_indices=True)   # nearest original non-cane pixel
        out[speck] = a[ri[speck], ci[speck]]
    return out


def change(a, b):
    c = np.zeros(a.shape, np.uint8)
    c[(a != 1) & (b == 1)] = 1; c[(a == 1) & (b != 1)] = 2
    return c


def acres(x): return {k: round(float((x == v).sum()) * AC, 1) for k, v in (("sugarcane_ac", 1), ("cassava_ac", 281), ("none_ac", 0))}


def write(f, arr, prof, qml):
    p = dict(prof, dtype=arr.dtype.name, nodata=ND if arr.dtype == np.uint16 else None, compress="deflate", predictor=2)
    with rasterio.open(f, "w", **p) as d: d.write(arr, 1)
    Path(f).with_suffix(".qml").write_text(qml)


def test(name):
    site = SP.site_dir(name); a, prof = load(site); o = site / "data/model/layer2_final_sieve_test"; o.mkdir(exist_ok=True)
    qml = (site / "data/model/layer2_final/class_2026.qml").read_text(); rows = [dict(site=site.name, size=0, **acres(a))]
    for size in map(int, os.environ.get("SIZES", "4,8,12,16,20,30").split(",")):
        b = run(a, size); c = change(a, b)
        write(o / f"class_s{size}.tif", b, prof, qml); write(o / f"change_s{size}.tif", c, prof, CHANGE_QML)
        rows.append(dict(site=site.name, size=size, **acres(b), gap_filled_ac=round((c == 1).sum() * AC, 1),
                         speck_removed_ac=round((c == 2).sum() * AC, 1), changed_pct=round(100 * (c > 0).sum() / (a != ND).sum(), 2)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(o / "sieve_test.csv", index=False); return rows


def apply(name):
    site = SP.site_dir(name); a, prof = load(site); size = int(os.environ.get("SIZE", 12))
    o = site / "data/model/layer2_final_sieve"; o.mkdir(exist_ok=True)
    b = run(a, size); c = change(a, b)
    write(o / "class_2026.tif", b, prof, (site / "data/model/layer2_final/class_2026.qml").read_text())
    write(o / "change_2026.tif", c, prof, CHANGE_QML)
    j = dict(site=site.name, src="layer2_final/class_2026.tif", size_px=size, rule="holes enclosed by cane only + isolated cane specks; 8-connected", before=acres(a), after=acres(b),
             gap_filled_ac=round((c == 1).sum() * AC, 1), speck_removed_ac=round((c == 2).sum() * AC, 1))
    (o / "sieve.json").write_text(json.dumps(j, indent=1)); return j


if __name__ == "__main__":
    mode, sites = sys.argv[1], sys.argv[2:]
    with ProcessPoolExecutor(min(len(sites), R.cpus())) as ex:
        for r in ex.map(test if mode == "test" else apply, sites): print(json.dumps(r) if mode == "apply" else f"{r[0]['site']} done")

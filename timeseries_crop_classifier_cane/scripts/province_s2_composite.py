"""Province S2 image for one month (user, 22 Oct 2026): per pixel the LATEST clear date of that month (Cloud Score+ cs_cdf 60-100),
from every district's band stacks, mosaicked into one province raster (all districts share the 10 m EPSG:32648 grid).
A pixel that is cloudy on every date of the month gets its least cloudy date (highest cs_cdf) so the image has no holes.
Bands: B2 B3 B4 B5 B8 (uint16, reflectance x 10000) + band 6 "date_MMDD" (e.g. 1012 = 12 Oct; 0 = outside the AOIs).
usage: python scripts/province_s2_composite.py <Province> <YYYYMM> [...]
  -> sites/2026/<Province>/<prov>_s2_<mon><year>.tif + .qml (5-3-2) + _dates.csv (area per date, share never clear)
"""
import multiprocessing as mp, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

import safe_resources as R
import site_paths as SP
from s2_date_image import QML

BANDS = ("B2", "B3", "B4", "B5", "B8")
MON = {"09": "sep", "10": "oct", "11": "nov", "06": "jun", "07": "jul", "08": "aug"}


def district(args):
    site, ym = args
    st = site / "data/stacks"
    with rasterio.open(st / "S2_cs_cdf.tif") as c:
        sel = [(i + 1, d) for i, d in enumerate(c.descriptions) if d.startswith(ym)]
        if not sel: return None
        sel.sort(key=lambda x: x[1], reverse=True)                     # newest first
        cs = c.read([i for i, _ in sel]).astype(np.int16); bounds = c.bounds
    with rasterio.open(site / "data/grid/index.tif") as g: aoi = g.read(2) > 0
    clear = (cs >= 60) & (cs <= 100)
    pick = np.where(clear.any(0), clear.argmax(0), np.where(cs <= 100, cs, -1).argmax(0))   # newest clear, else least cloudy
    out = np.zeros((6, *aoi.shape), np.uint16)
    for k, b in enumerate(BANDS):
        with rasterio.open(st / f"S2_{b}.tif") as r: a = r.read([i for i, _ in sel])
        out[k] = np.take_along_axis(a, pick[None], 0)[0]
    mmdd = np.array([int(d[4:]) for _, d in sel], np.uint16)
    out[5] = mmdd[pick]; out[:, ~aoi] = 0
    never = int((~clear.any(0) & aoi).sum())
    return site.name, bounds, out, never, int(aoi.sum())


def main(prov, ym):   # workers use spawn: fork after GDAL threads deadlocked on the 2nd month
    t0 = time.time(); folder = SP.SITES / "2026" / prov
    sites = [p for p in SP.all_sites("*_2026") if p.parent == folder and (p / "data/stacks/S2_cs_cdf.tif").exists()]
    with ProcessPoolExecutor(min(len(sites), R.cpus(), 3), mp_context=mp.get_context("spawn")) as ex: res = [r for r in ex.map(district, [(s, ym) for s in sites]) if r]
    with rasterio.open(sites[0] / "data/stacks/S2_B2.tif") as r: crs, res_m = r.crs, r.res[0]
    l = min(b.left for _, b, *_ in res); t = max(b.top for _, b, *_ in res)
    rr = max(b.right for _, b, *_ in res); bb = min(b.bottom for _, b, *_ in res)
    W, H = round((rr - l) / res_m), round((t - bb) / res_m)
    mos = np.zeros((6, H, W), np.uint16)
    for name, b, out, never, n in res:
        r0, c0 = round((t - b.top) / res_m), round((b.left - l) / res_m)
        w = mos[:, r0:r0 + out.shape[1], c0:c0 + out.shape[2]]
        take = (out[5] > 0) & ((w[5] == 0) | (out[5] > w[5]))             # overlap: inside an AOI, newer date wins
        w[:, take] = out[:, take]
    stem = f"{prov.lower()}_s2_{MON.get(ym[4:], ym[4:])}{ym[:4]}"
    f = folder / f"{stem}.tif"
    prof = dict(driver="GTiff", dtype="uint16", count=6, width=W, height=H, crs=crs, transform=Affine(res_m, 0, l, 0, -res_m, t),
                nodata=0, compress="deflate", predictor=2, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rasterio.open(f, "w", **prof) as d:
        d.write(mos)
        for k, n in enumerate(BANDS + ("date_MMDD",), 1): d.set_band_description(k, n)
    f.with_suffix(".qml").write_text(QML)
    v, c = np.unique(mos[5][mos[5] > 0], return_counts=True)
    tab = pd.DataFrame({"date": [f"{ym[:4]}{x:04d}" for x in v], "area_ac": (c * res_m * res_m / 4046.8564224).round(1),
                        "share_pct": (100 * c / c.sum()).round(1)})
    never = sum(r[3] for r in res); tot = sum(r[4] for r in res)
    tab.to_csv(folder / f"{stem}_dates.csv", index=False)
    print(f, f"{time.time() - t0:.0f} s"); print(tab.to_string(index=False)); print(f"cloudy on every date (least-cloudy date used): {100 * never / tot:.1f} %")


if __name__ == "__main__":
    for ym in sys.argv[2:]: main(sys.argv[1], ym)

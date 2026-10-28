"""Was the Sentinel-1 VH dip WATER or BARE SOIL? (user, 20 Oct 2026: a Site109 cane field cut in April showed a 2-month VH/VV
"dip" that was dry bare land, not flooding; the rice model took it as a paddy.)
For every pixel: dip periods = half-months Apr-Sep where the 15-day VH anomaly (vs the pixel's own median) <= -4 dB on either
track (s1_15d_2026nov.tif). Over all clear S2 dates inside those half-months (stacks B3 B4 B8 + cs_cdf >= 60):
  dip_nir_min   min NIR (B8)            x10000  (water: < 1200; bright bare: > 2500)
  dip_ndwi_max  max NDWI (B3-B8)/(B3+B8) x1000 (water: >= -100; dry bare: <= -250)
  dip_ndvi_med  median NDVI             x1000
  dip_n_clear   number of clear S2 dates in the dip periods (0 = no dip or all cloudy)
  dip_water     1 if a clear S2 date in a dip period looks like water (NDWI >= -0.10 or NIR <= 0.12 with NDVI < 0.35)
  dip_bare      1 if a clear S2 date in a dip period looks like bright bare soil (NIR >= 0.25 and NDVI < 0.30 and NDWI <= -0.25)
Output <site>/data/features/s2_on_dip_2026.tif (int16).   usage: python scripts/s2_on_dip.py <site> [more]"""
import sys, time
from pathlib import Path
import numpy as np, rasterio
import site_paths as SP
P = Path(__file__).resolve().parents[1]; BANDS = ["dip_nir_min", "dip_ndwi_max", "dip_ndvi_med", "dip_n_clear", "dip_water", "dip_bare"]
HM = [f"2026{m:02d}{h}" for m in range(4, 10) for h in "ab"]


def one(site):
    t0 = time.time(); S = SP.site_dir(site) / "data"
    with rasterio.open(S / "features/s1_15d_2026nov.tif") as r:
        n = list(r.descriptions); prof = r.profile.copy(); dip = np.zeros((len(HM),) + r.shape, bool)
        for orb in ("RO164_DES", "RO099_ASC"):
            vh = np.stack([r.read(n.index(f"{orb}_VH_{p}") + 1) for p in HM]).astype("float32"); vh[vh <= -32000] = np.nan; vh /= 100
            with np.errstate(all="ignore"): dip |= (vh - np.nanmedian(vh, 0)) <= -4
    with rasterio.open(S / "stacks/S2_cs_cdf.tif") as c: dates = list(c.descriptions)
    H, W = dip.shape[1:]; nir_min = np.full((H, W), 32000, "int16"); ndwi_max = np.full((H, W), -32000, "int16"); nclear = np.zeros((H, W), "int16")
    water = np.zeros((H, W), bool); bare = np.zeros((H, W), bool); ndvi_list = []
    src = {b: rasterio.open(S / f"stacks/S2_{b}.tif") for b in ("B3", "B4", "B8", "cs_cdf")}
    for i, d in enumerate(dates):
        if not ("20260401" <= d <= "20260930"): continue
        hm = d[:6] + ("a" if int(d[6:]) <= 15 else "b"); k = HM.index(hm)
        cs = src["cs_cdf"].read(i + 1); ok = dip[k] & (cs >= 60) & (cs <= 100)
        if not ok.any(): continue
        b3, b4, b8 = (src[b].read(i + 1).astype("float32") for b in ("B3", "B4", "B8")); ok &= b8 > 0
        with np.errstate(all="ignore"): ndwi = (b3 - b8) / (b3 + b8); ndvi = (b8 - b4) / (b8 + b4)
        nclear += ok; nir_min = np.where(ok, np.minimum(nir_min, b8.astype("int16")), nir_min); ndwi_max = np.where(ok, np.maximum(ndwi_max, (ndwi * 1000).astype("int16")), ndwi_max)
        water |= ok & ((ndwi >= -0.10) | ((b8 <= 1200) & (ndvi < 0.35))); bare |= ok & (b8 >= 2500) & (ndvi < 0.30) & (ndwi <= -0.25)
        ndvi_list.append(np.where(ok, ndvi, np.nan).astype("float16"))
    for s in src.values(): s.close()
    with np.errstate(all="ignore"): ndvi_med = np.nanmedian(np.stack(ndvi_list).astype("float32"), 0) if ndvi_list else np.full((H, W), np.nan, "float32")
    out = np.stack([np.where(nclear > 0, nir_min, -32768), np.where(nclear > 0, ndwi_max, -32768), np.where(np.isfinite(ndvi_med), ndvi_med * 1000, -32768), nclear, water, bare]).astype("int16")
    prof.update(count=len(BANDS), dtype="int16", nodata=-32768, compress="deflate", predictor=2)
    with rasterio.open(S / "features/s2_on_dip_2026.tif", "w", **prof) as d:
        d.write(out); [d.set_band_description(i, b) for i, b in enumerate(BANDS, 1)]
    print(f"{site}: dip px {int((nclear > 0).sum())}, water {int(water.sum())}, bare {int(bare.sum())}  {time.time() - t0:.0f} s", flush=True)


for s in sys.argv[1:]: one(s)
print("S2DIP_DONE")

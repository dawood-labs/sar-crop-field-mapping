"""Download Sentinel-1 and Sentinel-2 for the AOI straight from Earth Engine (no export tasks).

What it does
------------
1. Builds one fixed 10 m grid (EPSG:32648) around the AOI and writes the index raster
   data/grid/index.tif: band 1 = pixel_id (row * width + col), band 2 = in_aoi (1 inside the AOI polygon).
   Every downloaded file uses exactly this grid, so pixel (row, col) is the same place in every file.
2. Sentinel-1 GRD (IW, VV+VH, already calibrated and terrain-corrected to dB by Earth Engine):
   one file per (track, date), bands VV, VH, angle, float32, NaN = no data.
   Only the tracks that cover 100 % of the AOI (RO164 descending, RO099 ascending).
   Several slices of one track on one date are mosaicked into one image.
3. Sentinel-2 L2A (S2_SR_HARMONIZED): one file per date (the 4 tiles of the same orbit are mosaicked),
   bands B2, B3, B4, B5, B8 as uint16 surface reflectance x 10000, 0 = no data. All dates, cloudy or not.
   Plus a matching cloud file (uint16, 65535 = no data) with every cloud layer:
     cs, cs_cdf          Cloud Score+ x 100 (higher = clearer; cs_cdf >= 60 is a common "clear" cut)
     MSK_CLDPRB          ESA cloud probability 0-100 (20 m)
     SCL                 ESA scene classification (3 shadow, 4 veg, 5 bare, 6 water, 8/9 cloud, 10 cirrus ...)
     QA60                old ESA cloud bitmask (bit 10 opaque, bit 11 cirrus); EMPTY (0) since the 2022 baseline
     MSK_CLASSI_OPAQUE,  the bitmask layers that replaced QA60 (1 = flagged)
     MSK_CLASSI_CIRRUS
   Per-tile CLOUDY_PIXEL_PERCENTAGE goes to the manifest.
3b. --only s2x: extra S2 bands B6, B7, B8A (red-edge) and B11, B12 (SWIR) per date in data/raw/s2_extra/S2X_YYYYMMDD.tif
4. Writes manifests (one row per file: date, scenes, valid and clear fractions inside the AOI).

Images are pulled with ee.data.computePixels in 1024 x 1024 blocks, many blocks in parallel threads.
Each file is written to *.tmp and renamed when complete, so a re-run skips finished files (resume).

Run from anywhere:  python scripts/download.py [--only s1|s2] [--limit N]
"""
from __future__ import annotations

from pathlib import Path
import argparse, io, json, logging, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
REPO = PROJ          # reuse its Earth Engine auth + resource helpers
sys.path.insert(0, str(REPO / "src"))
from sar_pipeline.auth import init_ee                         # noqa: E402
from sar_pipeline.resources import detect_resources          # noqa: E402

START, END = "2025-09-01", "2026-10-10"                       # END is exclusive (Earth Engine filterDate)
S1_TRACKS = {164: "DESCENDING", 99: "ASCENDING"}              # the two tracks with 100 % AOI coverage
S1_BANDS = ["VV", "VH", "angle"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B8"]
S2X_BANDS = ["B6", "B7", "B8A", "B11", "B12"]       # red-edge + SWIR (20 m, resampled nearest to the 10 m grid)
S2_QA_BANDS = ["MSK_CLDPRB", "SCL", "QA60", "MSK_CLASSI_OPAQUE", "MSK_CLASSI_CIRRUS"]
CLOUD_BANDS = ["cs", "cs_cdf"] + S2_QA_BANDS
CRS, RES = "EPSG:32648", 10
BLOCK = 1024
AOI_SHP = PROJ / "data/aoi/aoi.shp"
CFG = {"_root": str(REPO), "auth": {"key_file": __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json"), "project": __import__("os").environ.get("EE_PROJECT", "")}}

log = logging.getLogger("download")


# ---------------------------------------------------------------- grid + index raster
def make_grid():
    import geopandas as gpd, rasterio
    from rasterio.features import rasterize
    aoi = gpd.read_file(AOI_SHP).to_crs(CRS)
    xmin, ymin, xmax, ymax = aoi.total_bounds
    x0, y0 = np.floor(xmin / RES) * RES, np.ceil(ymax / RES) * RES
    w, h = int(np.ceil((xmax - x0) / RES)), int(np.ceil((y0 - ymin) / RES))
    grid = {"crs": CRS, "res": RES, "x0": float(x0), "y0": float(y0), "width": w, "height": h,
            "transform": [RES, 0, float(x0), 0, -RES, float(y0)], "start": START, "end_exclusive": END}
    (PROJ / "data/grid/grid.json").write_text(json.dumps(grid, indent=2))
    tr = rasterio.transform.from_origin(x0, y0, RES, RES)
    idx = (np.arange(h, dtype=np.uint32)[:, None] * w + np.arange(w, dtype=np.uint32)[None, :])
    inside = rasterize([(g, 1) for g in aoi.geometry], out_shape=(h, w), transform=tr, dtype="uint8")
    prof = dict(driver="GTiff", width=w, height=h, count=2, dtype="uint32", crs=CRS, transform=tr,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(PROJ / "data/grid/index.tif", "w", **prof) as dst:
        dst.write(idx, 1); dst.write(inside.astype("uint32"), 2)
        dst.set_band_description(1, "pixel_id"); dst.set_band_description(2, "in_aoi")
    log.info("grid %d x %d px (%.0f acres in AOI)", w, h, inside.sum() * RES * RES / 4046.856)
    return grid, inside.astype(bool), aoi


# ---------------------------------------------------------------- Earth Engine pixel fetch
def fetch(image, bands, np_dtype, grid, fill):
    """Return a (bands, H, W) array of `image` on the grid, fetched in parallel-safe blocks."""
    import ee
    H, W = grid["height"], grid["width"]
    out = np.full((len(bands), H, W), fill, dtype=np_dtype)
    img = image.select(bands).unmask(fill)
    for r0 in range(0, H, BLOCK):
        for c0 in range(0, W, BLOCK):
            h, w = min(BLOCK, H - r0), min(BLOCK, W - c0)
            req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
                   "grid": {"dimensions": {"width": w, "height": h}, "crsCode": CRS,
                            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": grid["x0"] + c0 * RES,
                                                "shearY": 0, "scaleY": -RES, "translateY": grid["y0"] - r0 * RES}}}
            for attempt in range(6):
                try:
                    arr = ee.data.computePixels(req)
                    break
                except Exception as e:                         # rate limits / transient 5xx: back off and retry
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt + np.random.rand())
            for i, b in enumerate(bands):
                out[i, r0:r0 + h, c0:c0 + w] = arr[b]
    return out


def write_tif(path, arr, grid, bands, nodata, tags):
    import rasterio
    tmp = path.with_suffix(".tmp")
    prof = dict(driver="GTiff", width=grid["width"], height=grid["height"], count=arr.shape[0], dtype=arr.dtype,
                crs=CRS, transform=rasterio.transform.from_origin(grid["x0"], grid["y0"], RES, RES),
                nodata=nodata, compress="deflate", predictor=2 if arr.dtype.kind in "ui" else 3,
                tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        for i, b in enumerate(bands, 1):
            dst.set_band_description(i, b)
        dst.update_tags(**tags)
    os.replace(tmp, path)


# ---------------------------------------------------------------- job lists
def s1_jobs(aoi_ee):
    import ee
    jobs = []
    for ro, pss in S1_TRACKS.items():
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi_ee).filterDate(START, END)
               .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.eq("relativeOrbitNumber_start", ro))
               .filter(ee.Filter.eq("orbitProperties_pass", pss))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        info = col.aggregate_array("system:index").getInfo()
        dates = sorted({i.split("_")[4][:8] for i in info})     # S1X_IW_GRDH_1SDV_YYYYMMDDT...
        for d in dates:
            ids = [i for i in info if i.split("_")[4][:8] == d]
            jobs.append({"sensor": "s1", "track": f"RO{ro:03d}_{pss[:3]}", "date": d, "ids": ids,
                         "out": PROJ / f"data/raw/s1/RO{ro:03d}_{pss[:3]}/S1_RO{ro:03d}_{pss[:3]}_{d}.tif"})
    return jobs


def s2x_jobs(aoi_ee):
    """Extra S2 bands (red-edge, SWIR) as a separate file per date: data/raw/s2_extra/S2X_YYYYMMDD.tif."""
    return [{**j, "sensor": "s2x", "out": PROJ / f"data/raw/s2_extra/S2X_{j['date']}.tif"} for j in s2_jobs(aoi_ee)]


def s2_jobs(aoi_ee):
    import ee
    col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi_ee).filterDate(START, END)
    info = col.aggregate_array("system:index").getInfo()          # YYYYMMDDTHHMMSS_..._TxxXXX
    ccp = dict(zip(info, col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()))
    dates = sorted({i[:8] for i in info})
    return [{"sensor": "s2", "date": d, "ids": [i for i in info if i[:8] == d],
             "tile_cloud_pct": ";".join(f"{i[-6:]}:{ccp[i]:.1f}" for i in info if i[:8] == d),
             "out": PROJ / f"data/raw/s2/S2_{d}.tif", "out_cs": PROJ / f"data/raw/s2_cloud/S2CLOUD_{d}.tif"} for d in dates]


# ---------------------------------------------------------------- one job
def run_job(job, grid, inside):
    import ee
    t0 = time.time()
    if job["sensor"] == "s1":
        imgs = [ee.Image(f"COPERNICUS/S1_GRD/{i}").resample("bilinear") for i in job["ids"]]
        arr = fetch(ee.ImageCollection(imgs).mosaic(), S1_BANDS, np.float32, grid, -9999.0)
        arr[arr == -9999.0] = np.nan
        valid = np.isfinite(arr[1])[inside].mean()
        write_tif(job["out"], arr, grid, S1_BANDS, np.nan,
                  {"date": job["date"], "track": job["track"], "scenes": ",".join(job["ids"]), "units": "dB (VV, VH); degrees (angle)"})
        row = {"sensor": "s1", "track": job["track"], "date": job["date"], "n_scenes": len(job["ids"]),
               "valid_frac_aoi": round(float(valid), 4), "file": str(job["out"].relative_to(PROJ)), "scenes": ";".join(job["ids"])}
    elif job["sensor"] == "s2x":
        imgs = ee.ImageCollection([ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{i}") for i in job["ids"]])
        arr = fetch(imgs.mosaic(), S2X_BANDS, np.uint16, grid, 0)
        write_tif(job["out"], arr, grid, S2X_BANDS, 0,
                  {"date": job["date"], "scenes": ",".join(job["ids"]), "units": "surface reflectance x 10000"})
        row = {"sensor": "s2x", "track": "orbit18", "date": job["date"], "n_scenes": len(job["ids"]),
               "valid_frac_aoi": round(float((arr[3] > 0)[inside].mean()), 4),
               "file": str(job["out"].relative_to(PROJ)), "scenes": ";".join(job["ids"])}
    else:
        imgs = ee.ImageCollection([ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{i}") for i in job["ids"]])
        arr = fetch(imgs.mosaic(), S2_BANDS, np.uint16, grid, 0)
        cs_col = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filter(ee.Filter.inList("system:index", job["ids"]))
        cs_missing = cs_col.size().getInfo() < len(job["ids"])     # Cloud Score+ is published a few days after the image
        if cs_missing:                                             # cs / cs_cdf = 65535 (no data); precompute falls back to ESA masks
            log.warning("%s: Cloud Score+ not yet available, cs layers left empty (delete the S2CLOUD file and re-run later)", job["date"])
            cs = ee.Image.constant([65535, 65535]).rename(["cs", "cs_cdf"]).toUint16()
        else:
            cs = cs_col.mosaic().multiply(100).round().toUint16()
        cloud = cs.addBands(imgs.select(S2_QA_BANDS).mosaic().toUint16())
        csa = fetch(cloud, CLOUD_BANDS, np.uint16, grid, 65535)
        write_tif(job["out"], arr, grid, S2_BANDS, 0,
                  {"date": job["date"], "scenes": ",".join(job["ids"]), "units": "surface reflectance x 10000"})
        write_tif(job["out_cs"], csa, grid, CLOUD_BANDS, 65535,
                  {"date": job["date"], "tile_cloudy_pixel_pct": job["tile_cloud_pct"], "cs_missing": str(int(cs_missing)),
                   "units": "cs/cs_cdf = Cloud Score+ x100 (higher = clearer); MSK_CLDPRB 0-100; SCL ESA classes; QA60/MSK_CLASSI_* bitmasks"})
        valid = (arr[0] > 0)[inside]
        clear = ((csa[1] != 65535) & (csa[1] >= 60) & (arr[0] > 0))[inside]
        row = {"sensor": "s2", "track": "orbit18", "date": job["date"], "n_scenes": len(job["ids"]),
               "valid_frac_aoi": round(float(valid.mean()), 4), "clear_frac_aoi": round(float(clear.mean()), 4),
               "tile_cloudy_pixel_pct": job["tile_cloud_pct"],
               "file": str(job["out"].relative_to(PROJ)), "scenes": ";".join(job["ids"])}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["s1", "s2", "s2x"])
    ap.add_argument("--limit", type=int, help="process only the first N pending jobs (test)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(PROJ / "logs/download.log"), logging.StreamHandler()])
    import ee, pandas as pd
    init_ee(CFG, high_volume=True)
    t0 = time.time()
    grid, inside, aoi = make_grid()
    aoi_ee = ee.Geometry(aoi.to_crs(4326).geometry.unary_union.__geo_interface__)
    jobs = []
    if a.only in (None, "s1"):
        jobs += s1_jobs(aoi_ee)
    if a.only in (None, "s2"):
        jobs += s2_jobs(aoi_ee)
    if a.only in (None, "s2x"):
        jobs += s2x_jobs(aoi_ee)
    log.info("listing done in %.0f s: %d S1 files, %d S2 dates", time.time() - t0,
             sum(j["sensor"] == "s1" for j in jobs), sum(j["sensor"] == "s2" for j in jobs))
    pending = [j for j in jobs if not j["out"].exists() or (j["sensor"] == "s2" and not j["out_cs"].exists())]
    if a.limit:
        pending = pending[: a.limit]
    for j in pending:
        j["out"].parent.mkdir(parents=True, exist_ok=True)
    # network-bound: a few parallel jobs, each fetching its blocks sequentially (Earth Engine allows ~40 concurrent requests)
    workers = max(2, min(12, detect_resources().cpus * 2))
    log.info("%d files to download, %d parallel workers", len(pending), workers)
    rows, failed = [], []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(run_job, j, grid, inside): j for j in pending}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                rows.append(f.result())
                log.info("[%d/%d] %s %s %s ok (%.0f s)", k, len(pending), j["sensor"], j.get("track", ""), j["date"], rows[-1]["seconds"])
            except Exception as e:
                failed.append(j); log.error("[%d/%d] %s %s FAILED: %s", k, len(pending), j["sensor"], j["date"], str(e)[:300])
    man = PROJ / "data/manifest/download_manifest.csv"
    old = pd.read_csv(man) if man.exists() else pd.DataFrame()
    allrows = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if len(allrows):
        allrows = allrows.drop_duplicates(["sensor", "track", "date"], keep="last").sort_values(["sensor", "track", "date"])
        allrows.to_csv(man, index=False)
    log.info("done in %.1f min: %d ok, %d failed", (time.time() - t0) / 60, len(rows), len(failed))


if __name__ == "__main__":
    main()

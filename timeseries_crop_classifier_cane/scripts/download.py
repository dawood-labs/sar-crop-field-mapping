"""Download Sentinel-1 and Sentinel-2 for the test AOI (copied from the field-mapping project) straight from Earth Engine (no export tasks).

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

import argparse, io, json, logging, os, sys, time
import site_paths as SP
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests.adapters  # noqa: E402  urllib3 pool per host defaults to 10 -> "Connection pool is full" with 32 block threads
requests.adapters.HTTPAdapter.__init__.__defaults__ = (64, 64, 0, False)   # pool_connections, pool_maxsize, max_retries, pool_block

PROJ = SP.site_dir(os.environ["SC_ROOT"]) if os.environ.get("SC_ROOT") else Path(__file__).resolve().parents[1]   # SC_ROOT = another site (external test)
# Earth Engine service account (28 Oct: from env, no dependency on another project's folder): EE_KEY_FILE = key json,
# EE_PROJECT = cloud project; the defaults are the JupyterHub pod's key.
EE_KEY_FILE = os.environ.get("EE_KEY_FILE", "secrets/ee_service_account.json")
EE_PROJECT = os.environ.get("EE_PROJECT", __import__("os").environ.get("EE_PROJECT", ""))


def init_ee_from_env(high_volume: bool = True):
    import ee
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(EE_KEY_FILE).with_scopes(
        ["https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"])
    kw = dict(credentials=creds, project=EE_PROJECT)
    if high_volume:                                   # many computePixels / getInfo calls
        kw["opt_url"] = "https://earthengine-highvolume.googleapis.com"
    ee.Initialize(**kw)

# 2025 test season (user, 11 Oct 2026): previous harvest Dec 2024 - Apr 2025 -> regrowth -> standing cane before the
# 1 Dec 2025 crush (the shared 2025 cane map is dated 2025-11-29). The 2026 season is a second run with other dates.
START, END = os.environ.get("DL_START", "2024-09-01"), os.environ.get("DL_END", "2025-12-01")
MAX_AOI_CLOUD = float(os.environ.get("DL_MAX_CLOUD", "0.75"))     # skip an S2 date when > 75 % of the AOI is cloudy                       # END is exclusive (Earth Engine filterDate)
S1_TRACKS = {int(k): v for k, v in (x.split(":") for x in os.environ.get("DL_S1_TRACKS", "164:DESCENDING,99:ASCENDING").split(","))}   # DL_S1_TRACKS="62:DESCENDING,172:ASCENDING" for other districts (same env as export_gcs.py)
S1_BANDS = ["VV", "VH"]                                      # user: no incidence angle
S2_BANDS = os.environ.get("DL_S2_BANDS", "B2,B3,B4,B5,B8").split(",")   # chips only: B3,B4,B8
S2X_BANDS = ["B6", "B7", "B8A", "B11", "B12"]       # red-edge + SWIR (20 m, resampled nearest to the 10 m grid)
S2_QA_BANDS = [b for b in os.environ.get("DL_S2_QA", "MSK_CLDPRB,SCL,QA60,MSK_CLASSI_OPAQUE,MSK_CLASSI_CIRRUS").split(",") if b]
CS_BANDS = os.environ.get("DL_CS_BANDS", "cs,cs_cdf").split(",")      # "cs_cdf" alone = only the clear mask the pipeline uses
CLOUD_BANDS = CS_BANDS + S2_QA_BANDS
S2_DIR = os.environ.get("DL_S2_DIR", "s2")   # e.g. s2_b2 to fetch one missing band into its own folder
CRS, RES = "EPSG:32648", 10
BLOCK = int(os.environ.get("DL_BLOCK", "1024"))   # smaller blocks (e.g. 256) for thin / scattered AOIs
INSIDE = None                                     # AOI mask of the grid (set in main): blocks without AOI pixels are skipped
AOI_SHP = Path(os.environ.get("SC_AOI") or PROJ / "data/aoi/aoi_site119.shp")

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
    (PROJ / "data/grid").mkdir(parents=True, exist_ok=True)
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

    def one(rc):
        r0, c0 = rc; h, w = min(BLOCK, H - r0), min(BLOCK, W - c0)
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

    blocks = [(r0, c0) for r0 in range(0, H, BLOCK) for c0 in range(0, W, BLOCK)
              if INSIDE is None or INSIDE[r0:r0 + BLOCK, c0:c0 + BLOCK].any()]   # no AOI pixel in the block -> stays fill
    # DL_BLOCK_THREADS > 1: blocks of ONE file in parallel (big AOIs, e.g. a whole province); keep jobs x threads <= ~40
    with ThreadPoolExecutor(int(os.environ.get("DL_BLOCK_THREADS", "1"))) as ex: list(ex.map(one, blocks))
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
    # cloudy fraction of the AOI per date: Cloud Score+ cs_cdf < 0.6 over the date's mosaic (100 m); tile % if no CS+
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(aoi_ee).filterDate(START, END)
    def frac(d):
        d = ee.String(d); t0 = ee.Date.parse("yyyyMMdd", d); day = cs.filterDate(t0, t0.advance(1, "day"))
        m = day.select("cs_cdf").mosaic().lt(0.6).rename("c")
        v = m.reduceRegion(ee.Reducer.mean(), aoi_ee, 100, maxPixels=1e9).get("c")
        return ee.Algorithms.If(day.size().gt(0), v, -1)
    cf = {}
    n = int(os.environ.get("DL_CLOUD_BATCH", "0")) or len(dates)        # >0: dates per request (big / scattered AOIs hit the
    for k in range(0, len(dates), n):                                     # Earth Engine user memory limit in one request)
        part = dates[k:k + n]; cf.update(zip(part, ee.List(part).map(frac).getInfo()))
    for d in dates:
        if cf[d] is None or cf[d] < 0:               # no Cloud Score+: mean tile CLOUDY_PIXEL_PERCENTAGE
            cf[d] = float(np.mean([ccp[i] for i in info if i[:8] == d])) / 100
    keep = [d for d in dates if cf[d] <= MAX_AOI_CLOUD]
    log.info("S2: %d dates, %d with <= %.0f %% cloud over the AOI kept", len(dates), len(keep), 100 * MAX_AOI_CLOUD)
    (PROJ / "data/manifest").mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"date": dates, "aoi_cloud_frac": [round(cf[d], 3) for d in dates], "kept": [d in keep for d in dates]}) \
        .to_csv(PROJ / f"data/manifest/s2_cloud_{START[:4]}_{END[:4]}.csv", index=False)
    dates = keep
    return [{"sensor": "s2", "date": d, "ids": [i for i in info if i[:8] == d],
             "tile_cloud_pct": ";".join(f"{i[-6:]}:{ccp[i]:.1f}" for i in info if i[:8] == d),
             "out": PROJ / f"data/raw/{S2_DIR}/S2_{d}.tif", "out_cs": PROJ / f"data/raw/{S2_DIR}_cloud/S2CLOUD_{d}.tif"} for d in dates]


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
                  {"date": job["date"], "track": job["track"], "scenes": ",".join(job["ids"]), "units": "dB (VV, VH)"})
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
        cloud = cs.addBands(imgs.select(S2_QA_BANDS).mosaic().toUint16()) if S2_QA_BANDS else cs
        csa = fetch(cloud, CLOUD_BANDS, np.uint16, grid, 65535)
        write_tif(job["out"], arr, grid, S2_BANDS, 0,
                  {"date": job["date"], "scenes": ",".join(job["ids"]), "units": "surface reflectance x 10000"})
        write_tif(job["out_cs"], csa, grid, CLOUD_BANDS, 65535,
                  {"date": job["date"], "tile_cloudy_pixel_pct": job["tile_cloud_pct"], "cs_missing": str(int(cs_missing)),
                   "units": "cs/cs_cdf = Cloud Score+ x100 (higher = clearer); MSK_CLDPRB 0-100; SCL ESA classes; QA60/MSK_CLASSI_* bitmasks"})
        valid = (arr[0] > 0)[inside]
        k = CLOUD_BANDS.index("cs_cdf")
        clear = ((csa[k] != 65535) & (csa[k] >= 60) & (arr[0] > 0))[inside]
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
    init_ee_from_env(high_volume=True)
    tag = os.environ.get("DL_WORKLOAD_TAG")            # GEE workload tag: compute of this run shows under this label in Cloud Monitoring
    if tag:
        ee.data.setDefaultWorkloadTag(tag); log.info("workload tag %s", tag)
    t0 = time.time()
    global INSIDE
    grid, inside, aoi = make_grid()
    INSIDE = inside
    (PROJ / "data/manifest").mkdir(parents=True, exist_ok=True)
    g = aoi.geometry.union_all()
    tol = float(os.environ.get("DL_AOI_SIMPLIFY", "0"))                  # metres; only for the cloud % / scene search, not the grid
    if tol: g = g.buffer(tol).simplify(tol)
    aoi_ee = ee.Geometry(__import__("geopandas").GeoSeries([g], crs=aoi.crs).to_crs(4326).iloc[0].__geo_interface__)
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
        if "out_cs" in j:                              # cloud layers go to their own folder
            j["out_cs"].parent.mkdir(parents=True, exist_ok=True)
    # network-bound: a few parallel jobs, each fetching its blocks sequentially (Earth Engine allows ~40 concurrent requests)
    workers = int(os.environ.get("DL_WORKERS", "20"))      # network-bound; Earth Engine allows ~40 concurrent requests
    # 28 Oct: each job holds a full (bands, H, W) array (+ a copy when writing) -> cap the workers by the pod's safe memory
    # (a 14k x 16k gap grid = ~2.2 GB per S2 file; 8 workers took down a 15 GB pod)
    from safe_resources import safe_memory_bytes
    per_job = 3 * 6 * 2 * grid["height"] * grid["width"]                # ~6 uint16 bands, x3 for write copies / cloud layers
    req = workers
    workers = max(1, min(workers, int(safe_memory_bytes() // per_job)))
    if workers < req:                                 # fewer files at once -> more parallel blocks per file (same network load)
        bt = int(os.environ.get("DL_BLOCK_THREADS", "1"))
        os.environ["DL_BLOCK_THREADS"] = str(min(32, bt * req // workers))
        log.info("memory allows %d files at once (of %d): %s block threads per file", workers, req, os.environ["DL_BLOCK_THREADS"])
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

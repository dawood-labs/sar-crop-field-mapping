"""Sentinel-1 / Sentinel-2 acquisition through Google Cloud Storage (user rule, 14 Oct 2026): Earth Engine batch exports
to our GCS prefix, then a download to the pod. Nothing is pulled from Earth Engine directly (computePixels).

Same jobs, grid and file layout as scripts/download.py (one GeoTIFF per S1 track/date with bands VV, VH in dB; one S2
file per date with B2..B8 and one cloud file with cs, cs_cdf, MSK_CLDPRB, SCL, ...). The GCS copy is the raw backup.

usage (env SC_ROOT=sites/<site>  SC_AOI=<aoi.shp>  DL_START  DL_END  DL_S1_TRACKS="164:DESCENDING,26:ASCENDING"):
  python scripts/export_gcs.py submit [--only s1|s2] [--limit N]   start the export tasks that have no file on GCS yet
  python scripts/export_gcs.py status                              refresh task states (data/manifest/export_tasks.csv)
  python scripts/export_gcs.py fetch                               download finished files, normalise to the download.py
                                                                   format (band names, nodata, tags), write download_manifest.csv
Exports land in gs://<bucket>/<prefix>/<site>/data/raw/...  (bucket/prefix as in scripts/upload.py).
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
tracks_env = os.environ.get("DL_S1_TRACKS", "164:DESCENDING,99:ASCENDING")
import download as D  # noqa: E402  (reads SC_ROOT / SC_AOI / DL_START / DL_END from the environment)
import site_paths as SP

D.S1_TRACKS = {int(a): b for a, b in (x.split(":") for x in tracks_env.split(","))}
PROJ = D.PROJ.resolve()
BUCKET = os.environ.get("GCS_BUCKET", __import__("os").environ.get("GCS_BUCKET", ""))
PREFIX = os.environ.get("GCS_PREFIX", __import__("os").environ.get("GCS_PREFIX", ""))
SITE_REL = SP.flat_rel(PROJ)      # e.g. sites/site39_2026 (GCS keeps the flat layout; local folder is sites/<year>/<Province>/<site>)
MAN = PROJ / "data/manifest/export_tasks.csv"
DESC = f"sc_{PROJ.name}"
MAX_ACTIVE = int(os.environ.get("EE_MAX_ACTIVE", "400"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gcs_key(path):
    return f"{PREFIX}/{SITE_REL}/{Path(path).resolve().relative_to(PROJ).as_posix()}"


def client():
    from google.cloud import storage
    from sar_pipeline.auth import credentials
    return storage.Client(credentials=credentials(D.CFG), project=D.CFG["auth"]["project"])


def list_remote():
    return {b.name: b.size for b in client().list_blobs(BUCKET, prefix=f"{PREFIX}/{SITE_REL}/data/raw/")}


def export_image(img, out, fill, grid):
    import ee
    W, H = grid["width"], grid["height"]
    x0, y0 = grid["x0"], grid["y0"]
    region = ee.Geometry.Rectangle([x0, y0 - H * D.RES, x0 + W * D.RES, y0], D.CRS, False)
    key = gcs_key(out)
    task = ee.batch.Export.image.toCloudStorage(
        image=img.unmask(fill), description=f"{DESC}_{Path(out).stem}"[:100], bucket=BUCKET,
        fileNamePrefix=key[:-4], region=region, crs=D.CRS, crsTransform=[D.RES, 0, x0, 0, -D.RES, y0],
        maxPixels=1e10, fileFormat="GeoTIFF", formatOptions={"noData": fill})
    task.start()
    return task.id, key


def job_images(job):
    """(image, out path, fill) for every file of a job."""
    import ee
    if job["sensor"] == "s1":
        imgs = [ee.Image(f"COPERNICUS/S1_GRD/{i}").resample("bilinear") for i in job["ids"]]
        return [(ee.ImageCollection(imgs).mosaic().select(D.S1_BANDS).toFloat(), job["out"], -9999.0)]
    if job["sensor"] == "s2x":                                      # red-edge + SWIR bands, own folder data/raw/s2_extra
        imgs = ee.ImageCollection([ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{i}") for i in job["ids"]])
        return [(imgs.mosaic().select(D.S2X_BANDS).toUint16(), job["out"], 0)]
    imgs = ee.ImageCollection([ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{i}") for i in job["ids"]])
    cs_col = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filter(ee.Filter.inList("system:index", job["ids"]))
    cs_missing = cs_col.size().getInfo() < len(job["ids"])
    cs = (ee.Image.constant([65535, 65535]).rename(["cs", "cs_cdf"]).toUint16() if cs_missing
          else cs_col.mosaic().multiply(100).round().toUint16())
    cloud = cs.addBands(imgs.select(D.S2_QA_BANDS).mosaic().toUint16()) if D.S2_QA_BANDS else cs
    job["cs_missing"] = int(cs_missing)
    return [(imgs.mosaic().select(D.S2_BANDS).toUint16(), job["out"], 0), (cloud.select(D.CLOUD_BANDS), job["out_cs"], 65535)]


def jobs_and_grid(only):
    import ee
    D.init_ee(D.CFG)
    grid, inside, aoi = D.make_grid()
    aoi_ee = ee.Geometry(aoi.to_crs(4326).geometry.unary_union.__geo_interface__)
    jobs = []
    if only in (None, "s1"):
        jobs += D.s1_jobs(aoi_ee)
    if only in (None, "s2"):
        jobs += D.s2_jobs(aoi_ee)
    if only == "s2x":
        jobs += D.s2x_jobs(aoi_ee)
    (PROJ / "data/manifest").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: (";".join(v) if isinstance(v, list) else str(v)) for k, v in j.items()} for j in jobs]) \
        .to_csv(PROJ / "data/manifest/export_jobs.csv", index=False)
    return jobs, grid, inside


def read_man():
    return pd.read_csv(MAN, dtype=str) if MAN.exists() else pd.DataFrame(columns=["key", "task_id", "description", "state", "sensor", "track", "date", "scenes", "cs_missing", "submitted"])


def submit(a):
    import ee
    jobs, grid, _ = jobs_and_grid(a.only)
    remote = list_remote()
    man = read_man()
    have = set(man.key) | set(remote)
    todo = []
    for j in jobs:
        outs = [j["out"]] + ([j["out_cs"]] if "out_cs" in j else [])
        if all(gcs_key(o) in have for o in outs):
            continue
        todo.append(j)
    if a.limit:
        todo = todo[: a.limit]
    active = sum(o["metadata"]["state"] in ("PENDING", "RUNNING") for o in ee.data.listOperations())
    log(f"{len(jobs)} jobs, {len(todo)} to submit, {active} tasks already active on the project")
    rows = []
    for k, j in enumerate(todo, 1):
        while active >= MAX_ACTIVE:
            time.sleep(60)
            active = sum(o["metadata"]["state"] in ("PENDING", "RUNNING") for o in ee.data.listOperations())
        for img, out, fill in job_images(j):
            key = gcs_key(out)
            if key in have:
                continue
            for attempt in range(5):
                try:
                    tid, key = export_image(img, out, fill, grid); break
                except Exception as e:
                    if attempt == 4:
                        raise
                    log("submit retry:", str(e)[:120]); time.sleep(10 * (attempt + 1))
            rows.append({"key": key, "task_id": tid, "description": f"{DESC}_{Path(out).stem}"[:100], "state": "SUBMITTED",
                         "sensor": j["sensor"], "track": j.get("track", ""), "date": j["date"], "scenes": ";".join(j["ids"]),
                         "cs_missing": j.get("cs_missing", ""), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
            active += 1
        if k % 20 == 0:
            log(f"submitted {k}/{len(todo)} jobs")
            pd.concat([man, pd.DataFrame(rows)]).to_csv(MAN, index=False)
    pd.concat([man, pd.DataFrame(rows)]).to_csv(MAN, index=False)
    log(f"done: {len(rows)} export tasks started (manifest {MAN})")


def status(_a=None):
    import ee
    D.init_ee(D.CFG)
    man = read_man()
    ops = {o["name"].split("/")[-1]: o["metadata"] for o in ee.data.listOperations()}
    for i, r in man.iterrows():
        m = ops.get(r.task_id)
        if m:
            man.loc[i, "state"] = m["state"]
            if m["state"] == "FAILED":
                man.loc[i, "error"] = m.get("error", {}).get("message", "")[:200] if isinstance(m.get("error"), dict) else str(m.get("error", ""))[:200]
    # reconcile with GCS (20 Oct 2026): an export whose file is on GCS but which has no manifest row (a duplicate task of an
    # earlier, unrecorded submit finished it; or the watchdog cancelled a stuck twin) gets a SUCCEEDED row, so fetch takes it
    try:
        have = set(man.key); remote = list_remote(); jobs, _, _ = jobs_and_grid(None); added = []
        for j in jobs:
            for img, out, fill in job_images(j):
                key = gcs_key(out)
                if key in have or not any(k.startswith(key[:-4]) for k in remote):
                    continue
                added.append({"key": key, "task_id": "", "description": f"{DESC}_{Path(out).stem}"[:100], "state": "SUCCEEDED", "sensor": j["sensor"],
                              "track": j.get("track", ""), "date": j["date"], "scenes": ";".join(j["ids"]), "cs_missing": j.get("cs_missing", ""), "submitted": "reconciled"})
        if added:
            man = pd.concat([man, pd.DataFrame(added)], ignore_index=True); log(f"reconciled {len(added)} file(s) found on GCS without a manifest row")
    except Exception as e:
        log("reconcile skipped:", str(e)[:120])
    man.to_csv(MAN, index=False)
    c = man.state.value_counts().to_dict()
    log("task states:", c)
    return man


def normalise(local_tmp, out, sensor, row, grid, inside):
    """Rewrite the exported GeoTIFF in the download.py format and return the manifest row."""
    import rasterio
    with rasterio.open(local_tmp) as r:
        arr = r.read()
    if sensor == "s1":
        arr = arr.astype("float32"); arr[arr <= -9998] = np.nan
        D.write_tif(out, arr, grid, D.S1_BANDS, np.nan, {"date": row["date"], "track": row["track"], "scenes": row["scenes"].replace(";", ","), "units": "dB (VV, VH)"})
        return {"sensor": "s1", "track": row["track"], "date": row["date"], "n_scenes": row["scenes"].count(";") + 1,
                "valid_frac_aoi": round(float(np.isfinite(arr[1])[inside].mean()), 4), "file": str(out.resolve().relative_to(PROJ)), "scenes": row["scenes"]}
    arr = arr.astype("uint16")
    if sensor == "s2x":
        D.write_tif(out, arr, grid, D.S2X_BANDS, 0, {"date": row["date"], "scenes": row["scenes"].replace(";", ","), "units": "surface reflectance x 10000"})
        return {"sensor": "s2x", "track": "orbit18", "date": row["date"], "n_scenes": row["scenes"].count(";") + 1,
                "valid_frac_aoi": round(float((arr[3] > 0)[inside].mean()), 4), "file": str(out.resolve().relative_to(PROJ)), "scenes": row["scenes"]}
    if "S2CLOUD" in out.name:
        D.write_tif(out, arr, grid, D.CLOUD_BANDS, 65535, {"date": row["date"], "cs_missing": str(row.get("cs_missing", "")),
                    "units": "cs/cs_cdf = Cloud Score+ x100 (higher = clearer); MSK_CLDPRB 0-100; SCL ESA classes; QA60/MSK_CLASSI_* bitmasks"})
        return None
    D.write_tif(out, arr, grid, D.S2_BANDS, 0, {"date": row["date"], "scenes": row["scenes"].replace(";", ","), "units": "surface reflectance x 10000"})
    return {"sensor": "s2", "track": "orbit18", "date": row["date"], "n_scenes": row["scenes"].count(";") + 1,
            "valid_frac_aoi": round(float((arr[0] > 0)[inside].mean()), 4), "file": str(out.resolve().relative_to(PROJ)), "scenes": row["scenes"]}


def fetch(_a=None):
    import rasterio
    man = status()
    grid = json.loads((PROJ / "data/grid/grid.json").read_text())
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        inside = r.read(2).astype(bool)
    remote = list_remote()
    cl = client(); bucket = cl.bucket(BUCKET)
    done = man[man.state == "SUCCEEDED"]
    todo = []
    for _, r in done.iterrows():
        out = PROJ / Path(r.key).relative_to(f"{PREFIX}/{SITE_REL}")
        if out.exists():
            continue
        shards = [k for k in remote if k.startswith(r.key[:-4])]
        if not shards:
            continue
        todo.append((r, out, shards))
    log(f"{len(done)} finished tasks, {len(todo)} files to fetch")
    tmpdir = PROJ / "data/raw/_gcs_tmp"; tmpdir.mkdir(parents=True, exist_ok=True)

    def one(item):
        r, out, shards = item
        paths = []
        for k in shards:
            p = tmpdir / Path(k).name
            b = bucket.blob(k); b.download_to_filename(str(p))
            if p.stat().st_size != remote[k]:
                raise RuntimeError(f"size mismatch {k}")
            paths.append(p)
        if len(paths) > 1:                                   # Earth Engine split the image: mosaic the shards
            from rasterio.merge import merge
            srcs = [rasterio.open(p) for p in paths]; arr, _ = merge(srcs); [s.close() for s in srcs]
            tmp = tmpdir / (out.stem + "_merged.tif")
            with rasterio.open(paths[0]) as s0:
                prof = s0.profile.copy(); prof.update(width=arr.shape[2], height=arr.shape[1])
            with rasterio.open(tmp, "w", **prof) as d:
                d.write(arr)
            paths = [tmp]
        out.parent.mkdir(parents=True, exist_ok=True)
        row = normalise(paths[0], out, r.sensor, r, grid, inside)
        for p in paths:
            p.unlink(missing_ok=True)
        return row

    rows, failed = [], []
    with ThreadPoolExecutor(int(os.environ.get("DL_WORKERS", "16"))) as pool:
        futs = {pool.submit(one, it): it for it in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                rr = f.result()
                if rr:
                    rows.append(rr)
            except Exception as e:
                failed.append(futs[f][1].name); log("FAILED", futs[f][1].name, str(e)[:200])
            if k % 25 == 0:
                log(f"fetched {k}/{len(todo)}")
    manf = PROJ / "data/manifest/download_manifest.csv"
    old = pd.read_csv(manf) if manf.exists() else pd.DataFrame()
    allrows = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if len(allrows):
        allrows.drop_duplicates(["sensor", "track", "date"], keep="last").sort_values(["sensor", "track", "date"]).to_csv(manf, index=False)
    pending = int((man.state != "SUCCEEDED").sum())
    log(f"fetch done: {len(rows)} S1/S2 files written, {len(failed)} failed, {pending} tasks not finished yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "status", "fetch"])
    ap.add_argument("--only", choices=["s1", "s2", "s2x"])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    (PROJ / "logs").mkdir(parents=True, exist_ok=True)
    {"submit": submit, "status": status, "fetch": fetch}[a.cmd](a)

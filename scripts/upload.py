"""Upload the project data to GCS: gs://$GCS_BUCKET/$GCS_PREFIX/<same relative path>.

Uploads data/{aoi,grid,manifest,rain,raw,stacks}, scripts/ and reports/ in parallel threads.
A file whose blob already exists with the same size is skipped, so re-running only sends new or changed files.
"""
from __future__ import annotations

from pathlib import Path
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sar_pipeline.auth import credentials  # noqa: E402

BUCKET, PREFIX = __import__("os").environ.get("GCS_BUCKET", ""), __import__("os").environ.get("GCS_PREFIX", "")
INCLUDE = ["data/aoi", "data/ancillary", "data/features", "data/model", "data/labels_v1", "data/gcps", "data/grid", "data/manifest", "data/rain", "data/raw", "data/stacks", "scripts", "notebooks", "reports"]
CFG = {"_root": str(PROJ),
       "auth": {"key_file": __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json"), "project": __import__("os").environ.get("EE_PROJECT", "")}}


def main():
    from google.cloud import storage
    client = storage.Client(credentials=credentials(CFG), project=CFG["auth"]["project"])
    bucket = client.bucket(BUCKET)
    existing = {b.name: b.size for b in client.list_blobs(BUCKET, prefix=PREFIX + "/")}
    files = [f for d in INCLUDE for f in (PROJ / d).rglob("*") if f.is_file() and f.suffix not in (".tmp", ".pyc")]
    todo = [f for f in files if existing.get(f"{PREFIX}/{f.relative_to(PROJ)}") != f.stat().st_size]
    total = sum(f.stat().st_size for f in todo)
    print(f"{len(files)} files, {len(todo)} to upload ({total / 1e9:.2f} GB)", flush=True)
    t0 = time.time()

    def up(f):
        blob = bucket.blob(f"{PREFIX}/{f.relative_to(PROJ)}")
        blob.chunk_size = 32 * 1024 * 1024
        blob.upload_from_filename(str(f), timeout=600)
        return f.stat().st_size

    done = 0
    with ThreadPoolExecutor(16) as pool:                   # network-bound: many parallel streams
        for k, n in enumerate(pool.map(up, todo), 1):
            done += n
            if k % 25 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  {done / 1e9:.2f} GB  {done / 1e6 / max(time.time() - t0, 1):.0f} MB/s", flush=True)
    print(f"uploaded to gs://{BUCKET}/{PREFIX}/ in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

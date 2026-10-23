"""Back up project folders to GCS, verify every file by size, optionally delete the local copy.

Target: gs://$GCS_BUCKET/$GCS_PREFIX/<same relative path>   (defaults: our own prefix next to the shared source data)
usage: python scripts/upload.py data/raw [more paths ...] [--delete]
  --delete   remove the local files only if EVERY file exists on GCS with the same size (else nothing is deleted)
A file already on GCS with the same size is skipped; 32 parallel streams (network-bound, many files).
"""
from __future__ import annotations

import os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
EE_ROOT = os.environ.get("EE_HELPERS_ROOT", str(__import__("pathlib").Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.path.join(EE_ROOT, "scripts"))
from sar_pipeline.auth import credentials  # noqa: E402
import site_paths as SP

BUCKET = os.environ.get("GCS_BUCKET", __import__("os").environ.get("GCS_BUCKET", ""))
PREFIX = os.environ.get("GCS_PREFIX", __import__("os").environ.get("GCS_PREFIX", ""))
CFG = {"_root": EE_ROOT, "auth": {"key_file": os.environ.get("EE_KEY_FILE", __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json")),
                                  "project": os.environ.get("EE_PROJECT", __import__("os").environ.get("EE_PROJECT", ""))}}


def main():
    from google.cloud import storage
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    delete = "--delete" in sys.argv
    client = storage.Client(credentials=credentials(CFG), project=CFG["auth"]["project"])
    bucket = client.bucket(BUCKET)
    files = [f for a in args for f in ([SP.from_flat(a)] if SP.from_flat(a).is_file() else SP.from_flat(a).rglob("*"))
             if f.is_file() and f.suffix not in (".tmp", ".pyc")]
    existing = {b.name: b.size for b in client.list_blobs(BUCKET, prefix=PREFIX + "/")}
    key = lambda f: f"{PREFIX}/{SP.flat_rel(f)}"  # noqa: E731  (GCS keeps the flat sites/<site>/ layout)
    todo = [f for f in files if existing.get(key(f)) != f.stat().st_size]
    print(f"{len(files)} files ({sum(f.stat().st_size for f in files) / 1e9:.2f} GB), {len(todo)} to upload", flush=True)
    t0 = time.time()

    def up(f):
        b = bucket.blob(key(f)); b.chunk_size = 32 * 1024 * 1024
        b.upload_from_filename(str(f), timeout=900)
        return f.stat().st_size

    done = 0
    with ThreadPoolExecutor(32) as pool:
        for k, n in enumerate(pool.map(up, todo), 1):
            done += n
            if k % 50 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  {done / 1e9:.2f} GB  {done / 1e6 / max(time.time() - t0, 1):.0f} MB/s", flush=True)
    remote = {b.name: b.size for b in client.list_blobs(BUCKET, prefix=PREFIX + "/")}
    bad = [str(f.relative_to(PROJ)) for f in files if remote.get(key(f)) != f.stat().st_size]
    print(f"VERIFY gs://{BUCKET}/{PREFIX}/: {len(files)} files, {len(bad)} missing / size mismatch ({time.time() - t0:.0f} s)", flush=True)
    if bad:
        print("NOT deleting anything; first problems:", bad[:10]); sys.exit(1)
    if delete:
        for a in args:
            p = SP.from_flat(a)
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        print("local copy deleted:", args, flush=True)


if __name__ == "__main__":
    main()

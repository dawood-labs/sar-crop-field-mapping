"""Download a project path back from our GCS prefix (reverse of scripts/upload.py), size-verified, 32 streams.
usage: python scripts/fetch_raw.py sites/<site>/data/raw/s2 [more paths]"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
EE_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]); sys.path.insert(0, os.path.join(EE_ROOT, "scripts"))
from sar_pipeline.auth import credentials
from google.cloud import storage
import site_paths as SP
PROJ = Path(__file__).resolve().parents[1]; BUCKET = __import__("os").environ.get("GCS_BUCKET", ""); PREFIX = __import__("os").environ.get("GCS_PREFIX", "")
CFG = {"_root": EE_ROOT, "auth": {"key_file": __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json"), "project": __import__("os").environ.get("EE_PROJECT", "")}}
client = storage.Client(credentials=credentials(CFG), project=__import__("os").environ.get("EE_PROJECT", "")); bucket = client.bucket(BUCKET)
for a in sys.argv[1:]:
    blobs = list(client.list_blobs(BUCKET, prefix=f"{PREFIX}/{SP.flat_rel(SP.from_flat(a.rstrip('/')) if not Path(a).is_absolute() else a)}/"))
    loc = lambda b: SP.from_flat(b.name[len(PREFIX) + 1:])  # noqa: E731  (GCS flat sites/<site>/ -> local sites/<year>/<Province>/<site>/)
    todo = [b for b in blobs if not loc(b).exists() or loc(b).stat().st_size != b.size]
    print(f"{a}: {len(blobs)} files on GCS, {len(todo)} to download ({sum(b.size for b in todo) / 1e9:.2f} GB)", flush=True)
    t0 = time.time()
    def dl(b):
        p = loc(b); p.parent.mkdir(parents=True, exist_ok=True); b.download_to_filename(str(p))
        assert p.stat().st_size == b.size, b.name
    with ThreadPoolExecutor(int(os.environ.get("DL_WORKERS", 32))) as ex: list(ex.map(dl, todo))
    print(f"  done {time.time() - t0:.0f} s", flush=True)

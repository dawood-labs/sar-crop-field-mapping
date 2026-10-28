"""S3 = the single source of truth for what is done. A stage is done when its marker
<prefix>/_state/<stage>.json exists AND every file the marker lists is on S3 with the listed size (a half-finished upload,
a deleted object or a marker from an interrupted run never counts as done).

Uploads / downloads run with many parallel streams (boto3 TransferManager); sizes are verified after every upload.
Credentials: the standard AWS chain (instance / pod role, env vars, ~/.aws). AWS_ENV=<file with `export AWS_...=` lines> is
read for the temporary SSO credentials used on the JupyterHub pod.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]


def _load_aws_env():
    f = os.environ.get("AWS_ENV")
    if not f or not os.path.exists(f):
        return
    for line in open(f):
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[7:].split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


_load_aws_env()
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-1")
import boto3                                     # noqa: E402
from boto3.s3.transfer import TransferConfig     # noqa: E402
from botocore.config import Config               # noqa: E402

_s3 = None


def client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", config=Config(max_pool_connections=64, retries={"max_attempts": 10, "mode": "adaptive"}))
    return _s3


def split(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    b, _, k = uri[5:].partition("/")
    return b, k


def listing(prefix_uri: str) -> dict[str, int]:
    """{key relative to the prefix: size} for every object under the prefix."""
    b, k = split(prefix_uri.rstrip("/") + "/")
    out = {}
    for page in client().get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=k):
        for o in page.get("Contents", []):
            out[o["Key"][len(k):]] = o["Size"]
    return out


def read_json(uri: str) -> dict | None:
    b, k = split(uri)
    try:
        return json.loads(client().get_object(Bucket=b, Key=k)["Body"].read())
    except client().exceptions.NoSuchKey:
        return None


def write_json(uri: str, obj: dict):
    b, k = split(uri)
    client().put_object(Bucket=b, Key=k, Body=json.dumps(obj, indent=1).encode(), ContentType="application/json")


def _threads() -> int:
    from safe_resources import io_threads
    return io_threads()


def upload_dir(local: Path, prefix_uri: str, subdirs: list[str], exclude=()) -> dict[str, int]:
    """Upload local/<subdir>/** to prefix/<subdir>/**, skip files already there with the same size, verify every size after.
    Returns {relative path: size} of everything that is now on S3 (used as the marker's file list)."""
    files = []
    for s in subdirs:                                         # a sub-folder (all files below it) or a single file
        q = local / s
        files += [q] if q.is_file() else [p for p in q.rglob("*") if p.is_file()] if q.exists() else []
    files = [p for p in files if not any(p.match(e) for e in exclude)]
    have = listing(prefix_uri)
    b, k0 = split(prefix_uri.rstrip("/") + "/")
    todo = [p for p in files if have.get(str(p.relative_to(local))) != p.stat().st_size]
    cfg = TransferConfig(max_concurrency=8, multipart_chunksize=64 * 2**20)
    with ThreadPoolExecutor(_threads()) as ex:
        list(ex.map(lambda p: client().upload_file(str(p), b, k0 + str(p.relative_to(local)), Config=cfg), todo))
    have = listing(prefix_uri)
    bad = [p for p in files if have.get(str(p.relative_to(local))) != p.stat().st_size]
    if bad:
        raise RuntimeError(f"S3 verify failed for {len(bad)} of {len(files)} files under {prefix_uri} (e.g. {bad[0]})")
    return {str(p.relative_to(local)): p.stat().st_size for p in files}


def download_prefix(prefix_uri: str, local: Path, only: list[str] | None = None) -> int:
    """Copy prefix/** (or only prefix/<sub>/** for sub in `only`) to local, skipping files already present with the same size."""
    have = listing(prefix_uri)
    b, k0 = split(prefix_uri.rstrip("/") + "/")
    keys = [r for r in have if not r.startswith("_state/") and (not only or any(r.startswith(o.rstrip("/") + "/") or r == o for o in only))]
    todo = [r for r in keys if not (local / r).exists() or (local / r).stat().st_size != have[r]]
    def get(r):
        (local / r).parent.mkdir(parents=True, exist_ok=True)
        tmp = local / (r + ".part")
        client().download_file(b, k0 + r, str(tmp))
        os.replace(tmp, local / r)
    with ThreadPoolExecutor(_threads()) as ex:
        list(ex.map(get, todo))
    return len(todo)


def code_hash() -> str:
    """Hash of the pipeline code (scripts/*.py + scripts/pipeline/*.py): stored in every marker, so a result can be traced to code."""
    h = hashlib.sha1()
    for p in sorted(list((PROJ / "scripts").glob("*.py")) + list((PROJ / "scripts/pipeline").glob("*.py"))):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


class Stage:
    """Marker of one stage of one unit (district, gap, province, fd version) under <prefix>/_state/<name>.json."""

    def __init__(self, prefix_uri: str, name: str):
        self.prefix, self.name = prefix_uri.rstrip("/"), name
        self.uri = f"{self.prefix}/_state/{name}.json"

    def done(self, verify: bool = True) -> bool:
        m = read_json(self.uri)
        if m is None:
            return False
        if not verify or not m.get("files"):
            return True
        have = listing(self.prefix)
        return all(have.get(r) == s for r, s in m["files"].items())

    def mark(self, files: dict[str, int], t0: float, **extra):
        write_json(self.uri, {"stage": self.name, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "seconds": round(time.time() - t0), "host": socket.gethostname(), "code": code_hash(),
                              "n_files": len(files), "files": files, **extra})

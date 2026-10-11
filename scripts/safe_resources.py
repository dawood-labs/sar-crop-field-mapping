"""Memory / worker sizing that is safe on this Kubernetes JupyterHub pod.

The pod has a memory REQUEST (MEM_GUARANTEE, 16 GB) and a LIMIT (MEM_LIMIT, 60 GB). Using more than the request is
allowed, but when the node runs short of memory Kubernetes evicts the pods that are above their request first, and
the whole server restarts. So jobs are sized to stay well under the request, not the limit.
Import this module BEFORE rasterio: it also caps GDAL's block cache (default 5 % of the host RAM per process).
"""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_CACHEMAX", "256")          # MB per process


def safe_memory_bytes(fraction: float = 0.6) -> int:
    guarantee = int(os.environ.get("MEM_GUARANTEE", 0)) or None
    try:
        avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        avail = guarantee or 8 * 2**30
    return int(fraction * min(x for x in (guarantee, avail) if x))


def workers(per_worker_bytes: float, cap: int | None = None) -> int:
    n = max(1, int(safe_memory_bytes() // max(per_worker_bytes, 1)))
    return max(1, min(n, os.cpu_count() or 1, cap or 10**6))

"""Resource sizing that adapts to the machine / pod it runs on (user rule 20 Oct 2026: never hard-code cores or threads;
more cores on a bigger pod must speed things up automatically, and nothing may exceed the pod's memory REQUEST).

- cpus()              usable cores = CPU affinity ∩ cgroup quota (/sys/fs/cgroup/cpu.max) ∩ optional cap SC_CPUS
                      (a pipeline lane that shares the pod with another lane sets SC_CPUS to its share).
- safe_memory_bytes() 60 % of min(MEM_GUARANTEE, available RAM): the pod is evicted when it exceeds its request
                      (MEM_GUARANTEE, e.g. 16 or 36 GB), so jobs plan against the request, never the limit.
- workers(bytes)      min(cpus, memory / bytes per worker): processes or threads for CPU/RAM-bound raster work.
- io_threads()        threads for network-bound work (downloads / uploads): 4 x cpus, 8..64.
Import this module BEFORE rasterio: it also caps GDAL's block cache (default 5 % of the host RAM per process).
"""
from __future__ import annotations

import os

os.environ.setdefault("GDAL_CACHEMAX", "256")          # MB per process
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")  # multi-threaded GeoTIFF (de)compression: use every core


def cpus() -> int:
    n = os.cpu_count() or 1
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:                                                   # cgroup v2 quota: "800000 100000" = 8 cores; "max 100000" = no quota
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            n = min(n, max(1, int(int(q) / int(p) + 0.5)))
    except (OSError, ValueError):
        pass
    cap = os.environ.get("SC_CPUS")
    if cap and cap.isdigit():
        n = min(n, int(cap))
    return max(1, n)


def safe_memory_bytes(fraction: float = 0.6) -> int:
    guarantee = int(os.environ.get("MEM_GUARANTEE", 0)) or None
    return int(fraction * min(x for x in (guarantee, available_bytes()) if x))


def available_bytes() -> int:
    """RAM this process can still use: MemAvailable (free + reclaimable page cache; 28 Oct: SC_AVPHYS_PAGES counted only
    free pages, so a pod with 10 GB of cache looked like 1 GB) capped by the cgroup headroom (memory.max - memory.current)."""
    avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) * 1024; break
    except OSError:
        pass
    if avail is None:
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            avail = 8 * 2**30
    try:                                              # cgroup headroom WITHOUT the page cache (28 Oct: counting the cache of
        lim = open("/sys/fs/cgroup/memory.max").read().strip()   # a busy pod left ~2 GB -> every step ran with 1 worker)
        if lim != "max":
            used = int(open("/sys/fs/cgroup/memory.current").read())
            st = dict(l.split() for l in open("/sys/fs/cgroup/memory.stat") if l.startswith(("file ", "anon ")))
            avail = min(avail, int(lim) - (used - int(st.get("file", 0))))
    except (OSError, ValueError):
        pass
    return max(avail, 256 * 2**20)


def workers(per_worker_bytes: float, cap: int | None = None) -> int:
    n = max(1, int(safe_memory_bytes() // max(per_worker_bytes, 1)))
    return max(1, min(n, cpus(), cap or 10**6))


def io_threads() -> int:
    return max(8, min(64, 4 * cpus()))


def describe() -> str:
    return f"{cpus()} cpus (SC_CPUS={os.environ.get('SC_CPUS', '-')}), safe memory {safe_memory_bytes() / 2**30:.1f} GB (MEM_GUARANTEE={int(os.environ.get('MEM_GUARANTEE', 0)) / 2**30:.0f} GB)"

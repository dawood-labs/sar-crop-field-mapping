"""Runtime machine-resource detection.

Why this exists
---------------
The same code runs on a laptop (WSL) and on large cloud notebook servers (e.g. JupyterHub on AWS).
Nothing in the pipeline may hard-code a number of cores, threads or bytes of RAM. Every such
decision is derived here from what the machine actually offers *at runtime*.

Containers (JupyterHub, Docker, Kubernetes) are the tricky part: `os.cpu_count()` and
`/proc/meminfo` report the HOST machine (e.g. 96 cores / 384 GB) even when the container is
limited to 8 cores / 32 GB. Using host numbers would oversubscribe the container and get the
process killed (OOM). So we read the cgroup limits and take the tightest one:

- the process's own cgroup comes from /proc/self/cgroup (JupyterHub spawners often place each user
  in a nested cgroup such as /kubepods/pod…/container or /system.slice/jupyter-user.service);
- a limit set on any ANCESTOR cgroup also applies, so every level up to the root is checked;
- both cgroup v2 (unified) and v1 (per-controller) layouts are supported.

Memory "in use" by a cgroup includes the page cache (files recently read or downloaded). The kernel
frees that cache on demand, so the reclaimable part (`inactive_file`) is not counted as used;
otherwise, right after a large download, the available memory would look like zero.

Config (all optional, section `resources:`)
    cpus: auto | int            cap on usable CPUs
    memory_fraction: 0.6        share of AVAILABLE memory the pipeline may plan to use
    io_threads: auto | int      threads for network-bound work (downloads)
    cpu_workers: auto | int     processes for CPU/RAM-bound local raster work
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_UNLIMITED = 1 << 60  # cgroup v1 reports "no limit" as a huge number
_SYS_ROOT = Path("/")  # tests point this at a fake /proc + /sys tree


@dataclass(frozen=True)
class Resources:
    cpus: int                    # usable CPUs (affinity ∩ cgroup quota ∩ config cap)
    memory_total_bytes: int      # tightest container limit if any, else physical RAM
    memory_available_bytes: int  # what can be allocated right now (page cache counted as free)
    notes: str                   # where the numbers came from (for logs)

    def describe(self) -> str:
        gb = 1024**3
        return (
            f"{self.cpus} usable CPUs, {self.memory_available_bytes / gb:.1f} GB available "
            f"of {self.memory_total_bytes / gb:.1f} GB ({self.notes})"
        )


# ---------------------------------------------------------------- low-level readers
def _read(path: str) -> str | None:
    p = _SYS_ROOT / path.lstrip("/")
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _int_setting(value) -> int | None:
    """An explicit integer setting; bools and "auto" are not integers here."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _own_cgroups() -> tuple[str | None, dict[str, str]]:
    """(cgroup v2 path, {v1 controller: path}) of this process, from /proc/self/cgroup."""
    v2, v1 = None, {}
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            v2 = path
        else:
            for c in controllers.split(","):
                if c:
                    v1[c] = path
    return v2, v1


def _levels(base: str, rel: str | None) -> list[str]:
    """base/<rel>, base/<parent of rel>, ..., base  (own cgroup first, root last)."""
    parts = [p for p in (rel or "/").strip("/").split("/") if p]
    return ["/".join([base.rstrip("/"), *parts[:i]]) for i in range(len(parts), -1, -1)]


def _reclaim_aware_usage(current: str | None, stat: str | None, inactive_key: str) -> int | None:
    if current is None:
        return None
    used = int(current)
    for line in (stat or "").splitlines():
        key, _, val = line.partition(" ")
        if key == inactive_key:
            used -= int(val)
            break
    return max(0, used)


def _cgroup_cpu_limit() -> float | None:
    v2, v1 = _own_cgroups()
    limits = []
    for d in _levels("/sys/fs/cgroup", v2):
        val = _read(f"{d}/cpu.max")  # "max 100000" or "200000 100000"
        if val:
            quota, _, period = val.partition(" ")
            if quota != "max" and period:
                limits.append(int(quota) / int(period))
    for mount in ("/sys/fs/cgroup/cpu,cpuacct", "/sys/fs/cgroup/cpu"):
        for d in _levels(mount, v1.get("cpu")):
            quota, period = _read(f"{d}/cpu.cfs_quota_us"), _read(f"{d}/cpu.cfs_period_us")
            if quota and period and int(quota) > 0:
                limits.append(int(quota) / int(period))
    return min(limits) if limits else None


def _cgroup_memory() -> tuple[int | None, int | None]:
    """(tightest memory limit, smallest free headroom) over this cgroup and all its ancestors.

    Both matter on shared JupyterHub nodes: our own cgroup may allow 16 GB while the parent slice that
    all users share has only 1 GB left. Usage is reclaim-aware (page cache is not counted).
    Returns (None, None) when no level has a limit.
    """
    v2, v1 = _own_cgroups()
    levels: list[tuple[int, int | None]] = []
    for d in _levels("/sys/fs/cgroup", v2):
        limit = _read(f"{d}/memory.max")
        if limit and limit != "max":
            usage = _reclaim_aware_usage(_read(f"{d}/memory.current"), _read(f"{d}/memory.stat"), "inactive_file")
            levels.append((int(limit), usage))
    for d in _levels("/sys/fs/cgroup/memory", v1.get("memory")):
        limit = _read(f"{d}/memory.limit_in_bytes")
        if limit and int(limit) < _UNLIMITED:
            usage = _reclaim_aware_usage(_read(f"{d}/memory.usage_in_bytes"), _read(f"{d}/memory.stat"),
                                         "total_inactive_file")
            levels.append((int(limit), usage))
    if not levels:
        return None, None
    tightest = min(limit for limit, _ in levels)
    headroom = min(max(0, limit - (usage or 0)) for limit, usage in levels)
    return tightest, headroom


def _host_memory() -> tuple[int, int] | None:
    """(total, available) of the host/VM from /proc/meminfo, or psutil on other systems."""
    meminfo = _read("/proc/meminfo")
    if meminfo:
        vals = {}
        for line in meminfo.splitlines():
            key, _, rest = line.partition(":")
            if rest.split():
                vals[key] = int(rest.split()[0]) * 1024
        if "MemTotal" in vals:
            return vals["MemTotal"], vals.get("MemAvailable", vals.get("MemFree", vals["MemTotal"]))
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except ImportError:
        return None


def _usable_cpus() -> tuple[int, str]:
    try:
        n = len(os.sched_getaffinity(0))
        src = "affinity"
    except AttributeError:  # Windows / macOS
        n = os.cpu_count() or 1
        src = "os.cpu_count"
    quota = _cgroup_cpu_limit()
    if quota is not None and quota < n:
        return max(1, int(quota)), f"cgroup cpu quota {quota:g}"
    return n, src


# ---------------------------------------------------------------- public API
def detect_resources(cfg: dict | None = None) -> Resources:
    rcfg = (cfg or {}).get("resources", {}) or {}
    cpus, cpu_src = _usable_cpus()
    cap = _int_setting(rcfg.get("cpus"))
    if cap is not None:
        cpus = max(1, min(cpus, cap))
        cpu_src += f", capped by config to {cap}"

    host = _host_memory()
    cg_limit, cg_usage = _cgroup_memory()  # (tightest limit, smallest headroom)
    if host is None and cg_limit is None:
        total = avail = 4 * 1024**3
        mem_src = "memory unknown -> assumed 4 GB"
    else:
        total, avail = host if host else (cg_limit, cg_limit)
        mem_src = "host meminfo"
        if cg_limit is not None:
            if cg_limit < total:
                total = cg_limit
                mem_src = "cgroup memory limit"
            if cg_usage is not None and cg_usage < avail:  # cg_usage = smallest headroom over all levels
                avail = cg_usage
                mem_src = "cgroup memory headroom"
    return Resources(cpus=cpus, memory_total_bytes=int(total), memory_available_bytes=int(avail),
                     notes=f"{cpu_src}; {mem_src}")


def memory_budget_bytes(res: Resources, cfg: dict | None = None) -> int:
    frac = float(((cfg or {}).get("resources", {}) or {}).get("memory_fraction", 0.6))
    return max(64 * 1024**2, int(res.memory_available_bytes * frac))


def io_threads(res: Resources, cfg: dict | None = None, cap: int = 32) -> int:
    """Threads for network-bound work. Network waits dominate, so more threads than CPUs is fine."""
    val = _int_setting(((cfg or {}).get("resources", {}) or {}).get("io_threads"))
    if val is not None:
        return max(1, val)
    return max(2, min(cap, res.cpus * 4))


def cpu_workers(res: Resources, cfg: dict | None = None, mem_per_worker_bytes: int | None = None) -> int:
    """Processes for CPU/RAM-bound work: leave one CPU for the notebook/OS, and never exceed the RAM budget."""
    val = _int_setting(((cfg or {}).get("resources", {}) or {}).get("cpu_workers"))
    n = val if val is not None else (res.cpus - 1 if res.cpus > 2 else res.cpus)
    if mem_per_worker_bytes:
        n = min(n, memory_budget_bytes(res, cfg) // mem_per_worker_bytes)
    return max(1, int(n))


def submit_threads(res: Resources, cfg: dict | None = None) -> int:
    """Threads for Earth Engine task submission. Bound by the EE API rate limit, not local hardware."""
    val = _int_setting(((cfg or {}).get("export", {}) or {}).get("max_workers"))
    if val is not None:
        return max(1, val)
    return max(2, min(8, res.cpus * 2))


def block_rows(res: Resources, width: int, n_bands: int, bytes_per_value: int,
               workers: int = 1, cfg: dict | None = None, copies: float = 3.0) -> int:
    """How many raster rows one worker may read at once.

    `copies` accounts for temporary arrays (masks, dtype casts) created while processing a block.
    """
    per_row = max(1, width * n_bands * bytes_per_value * copies)
    rows = memory_budget_bytes(res, cfg) // max(1, workers) // per_row
    return max(1, int(rows))


def gdal_cache_bytes(res: Resources, fraction: float = 0.1) -> int:
    """GDAL block cache size (GDAL_CACHEMAX); small share of available RAM, at least 64 MB.

    Callers that also plan memory with `memory_budget_bytes` should count this inside that budget.
    """
    return max(64 * 1024**2, int(res.memory_available_bytes * fraction))


def disk_free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free

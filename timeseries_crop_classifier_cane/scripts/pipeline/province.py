"""Province steps, resumable from S3 (each needs every site of the province done on S3 first).

  outputs   mosaic of all sites (districts / gap tiles / grid tiles): layer 1 cane map + layer 2 classes -> clean (sieve 20 px)
            -> clip to the official province polygon -> sugarcane polygons (area_ac) -> S3 <province>/province/ (+ marker)
  fd <v>    field-delineation sugarcane polygons, one version per task (v1 | 1ac | 1p5ac | 2ac) -> S3 <province>/fd_review/
            needs the delineation fields + boundary on S3 (<delineation_root>/<Province>/) and the province class raster.
"""
from __future__ import annotations

import os
import shutil
import time

from . import s3state as S3
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import site_paths as SP  # noqa: E402
from .config import PROJ, Params, Province
from .site import ensure_shared, log, run, site_prefix

FD_ARG = {"v1": "-1", "1ac": "1", "1p5ac": "1.5", "2ac": "2"}


def finished_units(p: Params, prov: Province) -> tuple[list[str], list[str]]:
    """(units with a valid process marker: districts / grid tiles / gap tiles, sites still missing)."""
    done, missing = [], []
    for s in prov.sites:
        sp = site_prefix(p, s.name)
        m = S3.read_json(f"{sp}/_state/process.json")
        if m is None or not S3.Stage(sp, "process").done():
            missing.append(s.name); continue
        done += m.get("tiles") or [s.name]
    return done, missing


def work_dir(p: Params, what: str) -> Path:
    """Scratch space on the pod for one province step (S3 holds every input and output; this folder is deleted after)."""
    return PROJ / "_work" / p.year / p.province / what


def outputs(p: Params, prov: Province):
    pre = p.prefix()
    st = S3.Stage(f"{pre}/province", "outputs")
    if st.done() and not p.forced("province"):
        log(f"{p.province} province outputs: done on S3 - skipped"); return
    units, missing = finished_units(p, prov)
    if missing:
        raise RuntimeError(f"{p.province}: {len(missing)} sites not finished on S3: {', '.join(missing)}")
    t0 = time.time(); ensure_shared(p)
    stem = p.province.lower()
    work = work_dir(p, "province")                   # exactly the province's units, fetched from S3 (small outputs only)
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    for u in units:
        S3.download_prefix(site_prefix(p, u), work / u,
                           only=["data/model/layer2_final", "data/model/s2on_r2_oct_rn5", "data/grid", "data/aoi"])
    log(f"{len(units)} units fetched from S3")
    env = dict(os.environ, SC_SITES_PROVINCE_DIR=str(work))
    run(f"cd scripts && python province_mosaic.py {p.province} data/model/s2on_r2_oct_rn5/cane_map_2026.tif {stem}_cane_map_2026.tif "
        f"&& python province_cane.py {p.province}", env, work / "logs/province.log")
    outs = [f.name for f in work.glob(f"{stem}_*.*") if f.is_file()]
    files = S3.upload_dir(work, f"{pre}/province", outs + ["logs"])
    st.mark({k: v for k, v in files.items() if not k.startswith("logs/")}, t0, units=units)
    if not p.keep_local:
        shutil.rmtree(work, ignore_errors=True)
    log(f"{p.province} province outputs: done in {(time.time() - t0) / 60:.0f} min ({len(outs)} files)")


def fd(p: Params, version: str, cleanup: bool = False):
    """One FD version. Inputs (fields + boundary + class raster) and the version-independent cache (dedupe + labels + cuts,
    made by the first version) are fetched from / kept on S3; the scratch folder is reused by the next version on the same
    pod and deleted with cleanup=True (the last version of a run)."""
    pre = p.prefix()
    st = S3.Stage(f"{pre}/fd_review", f"fd_{version}")
    if st.done() and not p.forced("fd"):
        log(f"{p.province} fd {version}: done on S3 - skipped")
        if cleanup and not p.keep_local: shutil.rmtree(work_dir(p, "fd"), ignore_errors=True)
        return
    if not S3.Stage(f"{pre}/province", "outputs").done():
        raise RuntimeError(f"{p.province}: province outputs not on S3 yet (run the province step first)")
    t0 = time.time()
    stem = p.province.lower()
    W = work_dir(p, "fd"); D = W / "inputs"; O = W / "fd_review"; O.mkdir(parents=True, exist_ok=True)
    S3.download_prefix(f"{p.delineation_root.rstrip('/')}/{p.province}", D)
    S3.download_prefix(f"{pre}/province", D, only=[f"{stem}_class_2026_s20.tif"])
    S3.download_prefix(f"{pre}/fd_review", O, only=["_cache"])          # shared stage made by an earlier version (any pod)
    F = next(D.glob("*_fields_2026.parquet"), None); B = next(D.glob("*Boundary.gpkg"), None)
    if F is None or B is None:
        raise RuntimeError(f"{p.province}: need *_fields_2026.parquet and *Boundary.gpkg under {p.delineation_root}/{p.province}/")
    C = D / f"{stem}_class_2026_s20.tif"
    o = O / f"{stem}_fd_sugarcane_2026_{version}.gpkg"
    run(f'cd scripts && FD_CACHE_DIR="{O / "_cache"}" RESUME=1 NONCANE_MAX_AC={FD_ARG[version]} NONCANE_INCL_CASSAVA=1 '
        f'python -u fd_cane_fields.py "{F}" "{B}" "{C}" "{o}"', dict(os.environ), W / f"logs/fd_{version}.log")
    keep = [x.name for x in O.glob(f"{o.stem}*") if x.is_file() and "precleaned" not in x.name and "checkpoint" not in x.name]
    files = S3.upload_dir(O, f"{pre}/fd_review", keep)
    S3.upload_dir(O, f"{pre}/fd_review", ["_cache"]); S3.upload_dir(W, f"{pre}/fd_review", ["logs"])
    st.mark(files, t0)
    for x in O.glob(f"{o.stem}*"):                   # the version's own files are on S3 now
        x.unlink()
    if cleanup and not p.keep_local:
        shutil.rmtree(W, ignore_errors=True)
    log(f"{p.province} fd {version}: done in {(time.time() - t0) / 60:.0f} min")

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


def outputs(p: Params, prov: Province):
    pre = p.prefix()
    st = S3.Stage(f"{pre}/province", "outputs")
    if st.done() and not p.forced("province"):
        log(f"{p.province} province outputs: done on S3 - skipped"); return
    units, missing = finished_units(p, prov)
    if missing:
        raise RuntimeError(f"{p.province}: {len(missing)} sites not finished on S3: {', '.join(missing)}")
    t0 = time.time(); ensure_shared(p)
    folder = SP.SITES / p.year / p.province
    stem = p.province.lower()
    # the province scripts mosaic every site folder of the province that holds outputs -> a clean folder with only the units
    work = SP.SITES / p.year / f"_province_{p.province}"
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    for u in units:                                   # only the small outputs the province steps read
        S3.download_prefix(site_prefix(p, u), folder / u,
                           only=["data/model/layer2_final", "data/model/s2on_r2_oct_rn5", "data/grid", "data/aoi"])
        os.symlink(folder / u, work / u)
    log(f"{len(units)} units ready")
    env = dict(os.environ, SC_SITES_PROVINCE_DIR=str(work))
    lf = folder / "logs/province.log"
    run(f"cd scripts && python province_mosaic.py {p.province} data/model/s2on_r2_oct_rn5/cane_map_2026.tif {stem}_cane_map_2026.tif "
        f"&& python province_cane.py {p.province}", env, lf)
    outs = [f for f in work.glob(f"{stem}_*.*") if f.is_file()]
    for f in outs:                                    # results next to the sites, as on the JupyterHub pod
        shutil.move(str(f), folder / f.name)
    files = S3.upload_dir(folder, f"{pre}/province", [f.name for f in outs])
    st.mark(files, t0, units=units)
    shutil.rmtree(work, ignore_errors=True)
    log(f"{p.province} province outputs: done in {(time.time() - t0) / 60:.0f} min ({len(files)} files)")


def fd(p: Params, version: str):
    pre = p.prefix()
    st = S3.Stage(f"{pre}/fd_review", f"fd_{version}")
    if st.done() and not p.forced("fd"):
        log(f"{p.province} fd {version}: done on S3 - skipped"); return
    if not S3.Stage(f"{pre}/province", "outputs").done():
        raise RuntimeError(f"{p.province}: province outputs not on S3 yet (run the province step first)")
    t0 = time.time()
    stem = p.province.lower()
    folder = SP.SITES / p.year / p.province
    D = PROJ / "data/delineation" / f"{stem}_{p.year}"
    n = S3.download_prefix(f"{p.delineation_root.rstrip('/')}/{p.province}", D)
    S3.download_prefix(f"{pre}/province", folder, only=[f"{stem}_class_2026_s20.tif"])
    F = next(D.glob("*_fields_2026.parquet"), None); B = next(D.glob("*Boundary.gpkg"), None)
    if F is None or B is None:
        raise RuntimeError(f"{p.province}: need *_fields_2026.parquet and *Boundary.gpkg under {p.delineation_root}/{p.province}/")
    C = folder / f"{stem}_class_2026_s20.tif"
    O = folder / "fd_review"; O.mkdir(exist_ok=True)
    o = O / f"{stem}_fd_sugarcane_2026_{version}.gpkg"
    run(f'cd scripts && RESUME=1 NONCANE_MAX_AC={FD_ARG[version]} NONCANE_INCL_CASSAVA=1 python -u fd_cane_fields.py "{F}" "{B}" "{C}" "{o}"',
        dict(os.environ), folder / f"logs/fd_{version}.log")
    keep = [x.name for x in O.glob(f"{o.stem}*") if x.is_file() and "precleaned" not in x.name and "checkpoint" not in x.name]
    files = S3.upload_dir(O, f"{pre}/fd_review", keep)
    st.mark(files, t0)
    log(f"{p.province} fd {version}: done in {(time.time() - t0) / 60:.0f} min")

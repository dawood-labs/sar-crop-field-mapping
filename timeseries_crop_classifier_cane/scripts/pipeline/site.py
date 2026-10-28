"""One site (district, gap AOI or grid tile) end to end, resumable from S3.

Stages (each one skipped when its S3 marker is valid, unless forced):
  download  S1 VV/VH (2 tracks) + S2 B2 B3 B4 B5 B8 + cs_cdf from Earth Engine (direct read) -> data/raw -> S3 + marker
  process   stacks -> S1 15-day composites -> S2 dry features -> phenology -> layer 1 (s2on_r2 model + October bare rule +
            recall rule v5) -> layer 2 (colour v8, 9x9 smoothing) -> S3 (stacks, features, model outputs) + marker
            Gap AOIs (big sparse grids) are split into tiles after the stacks; every tile is processed and uploaded on its own.
A new pod resumes from S3: a finished download is fetched from S3 instead of Earth Engine; a finished process is skipped.
The science commands are the ones of the JupyterHub drivers (sites/_drivers/province_run_v2.sh, gap_run.sh): outputs are
bit-identical.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import s3state as S3
from .config import PROJ, Params, Site

sys.path.insert(0, str(PROJ / "scripts"))
import site_paths as SP          # noqa: E402

UPLOAD_PROCESS = ["data/stacks", "data/aoi", "data/grid", "data/manifest", "data/model", "data/features"]
UPLOAD_DOWNLOAD = ["data/raw", "data/aoi", "data/grid", "data/manifest"]
LAYER2 = "data/model/layer2_final/class3_2026.tif"
DONE_LOCAL = "data/model/.process_complete"
LAYER1_LOCAL, COLOUR_LOCAL = "data/model/s2on_r2/model.joblib", "sites/layer2/colour_model_v8.joblib"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def site_prefix(p: Params, site_name: str) -> str:
    return f"{p.prefix()}/districts/{site_name}"


def site_root(p: Params, site_name: str) -> Path:
    return SP.SITES / p.year / p.province / site_name


def run(cmd: str, env: dict, logfile: Path):
    """Run one pipeline command in the project root; its output goes to the site's log; raise on failure."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%d %H:%M:%S')} {cmd}\n"); f.flush()
        r = subprocess.run(cmd, shell=True, cwd=PROJ, env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        tail = "".join(open(logfile).readlines()[-15:])
        raise RuntimeError(f"failed ({r.returncode}): {cmd}\n--- last lines of {logfile} ---\n{tail}")


def free_gb() -> float:
    return shutil.disk_usage(PROJ).free / 2**30


def wait_for_disk(p: Params):
    """Stop before a step that would fill the disk: wait up to disk_wait_s (shared pod), else fail with a clear message."""
    t_end = time.time() + p.disk_wait_s
    while free_gb() < p.min_free_gb:
        if time.time() > t_end:
            raise RuntimeError(f"only {free_gb():.0f} GB free, need {p.min_free_gb:.0f} GB (bigger volume or lower min_free_gb)")
        time.sleep(60)


def fetch(uri: str, local: Path, mode: int | None = None):
    if local.exists():
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    b, k = S3.split(uri)
    S3.client().download_file(b, k, str(local) + ".part"); os.replace(str(local) + ".part", local)
    if mode:
        os.chmod(local, mode)


def ensure_shared(p: Params):
    """Models + source tables every site needs, fetched from S3 on a fresh pod (custom models get their own local name)."""
    for rel, uri in p.extra["shared_inputs"].items():
        if rel == LAYER1_LOCAL and p.layer1_model != uri:
            continue
        if rel == COLOUR_LOCAL and p.colour_model != uri:
            continue
        fetch(uri, PROJ / rel)
    if p.layer1_model != p.extra["shared_inputs"][LAYER1_LOCAL]:
        fetch(p.layer1_model, PROJ / "data/model/custom/model.joblib")
    if p.colour_model != p.extra["shared_inputs"][COLOUR_LOCAL]:
        fetch(p.colour_model, PROJ / "sites/layer2/colour_model_custom.joblib")
    if os.environ.get("EE_KEY_FILE", "").startswith("s3://"):    # key kept as an S3 object (Airflow: only S3 access)
        local = Path.home() / ".config/standing_cane/ee_key.json"
        fetch(os.environ["EE_KEY_FILE"], local, 0o600)
        os.environ["EE_KEY_FILE"] = str(local)


def ensure_aoi(p: Params, site: Site, root: Path):
    if (root / "data/aoi/aoi.shp").exists():
        return
    S3.download_prefix(site_prefix(p, site.name), root, only=["data/aoi"])
    if not (root / "data/aoi/aoi.shp").exists():
        raise RuntimeError(f"no AOI for {site.name} on S3: run `prepare --province {p.province}` first")


def process_cmds(p: Params) -> list[str]:
    model = "data/model/custom/model.joblib" if p.layer1_model != p.extra["shared_inputs"][LAYER1_LOCAL] else LAYER1_LOCAL
    ctag = "custom" if p.colour_model != p.extra["shared_inputs"][COLOUR_LOCAL] else "v8"
    return [
        "python -u scripts/s1_composites.py 2026nov",
        "python -u scripts/s2_dry_features.py 2026 {S}",
        "python -u scripts/pheno_features.py 2026 {S}",
        f"MODEL={model} OUT_TAG=s2on_r2 FEATURES_FILE=s1_15d_2026nov.tif YEAR=2026 python scripts/apply_site.py {{S}}",
        "python scripts/oct_bare_rule.py s2on_r2 s2on_r2_oct {S}",
        "RULE=5 python scripts/recall_rule.py s2on_r2_oct s2on_r2_oct_rn5 {S}",
        f"COLOUR_TAG={ctag} OUT_TAG=layer2_final SMOOTH=9 LAYER1=s2on_r2_oct_rn5 python scripts/colour_multi.py apply {{name}}",
    ]


def steps(p: Params) -> list[tuple[str, str, list[str]]]:
    """(name, command, outputs) of the process stage: every step's outputs go to S3 with their own marker
    (<prefix>/_state/step_<name>.json), so a new pod resumes at the first unfinished step (user, 28 Oct 2026)."""
    c = process_cmds(p)
    return [("stacks", "python -u scripts/build_stacks.py", ["data/stacks"]),
            ("s1_15d", c[0], ["data/features/s1_15d_2026nov*"]),
            ("s2_dry", c[1], ["data/features/s2_dry_2026*"]),
            ("pheno", c[2], ["data/features/pheno_2026*"]),
            ("layer1", c[3], ["data/model/s2on_r2"]),
            ("oct_rule", c[4], ["data/model/s2on_r2_oct"]),
            ("recall_rule", c[5], ["data/model/s2on_r2_oct_rn5"]),
            ("layer2", c[6], ["data/model/layer2_final"])]


def _expand(root: Path, pats: list[str]) -> list[str]:
    out = []
    for pat in pats:
        out += [str(x.relative_to(root)) for x in sorted(root.glob(pat))] if "*" in pat else ([pat] if (root / pat).exists() else [])
    return out


def run_steps(p: Params, root: Path, name: str, env: dict, prefix: str):
    """Each step: done on S3 -> skipped (its outputs fetched from S3 only if a later step needs them on this pod);
    else run -> upload its outputs (size-verified) -> marker. A killed step simply runs again."""
    lf = root / "logs/process.log"
    todo = [(n, c, o) for n, c, o in steps(p) if p.forced("process") or not S3.Stage(prefix, f"step_{n}").done()]
    if not todo:
        return
    first = [n for n, *_ in steps(p)].index(todo[0][0])
    need = [o for _, _, outs in steps(p)[:first] for o in outs]           # outputs of the finished steps the rest reads
    import fnmatch
    have = S3.listing(prefix)
    want = []                                                            # S3 keys of those outputs (+ grid / aoi / manifest)
    for pat in need + ["data/grid", "data/aoi", "data/manifest"]:
        want += [k for k in have if (fnmatch.fnmatch(k, pat) if "*" in pat else k.startswith(pat + "/"))]
    missing = [k for k in want if not (root / k).exists() or (root / k).stat().st_size != have[k]]
    if missing:
        n = S3.download_prefix(prefix, root, only=missing)
        log(f"process: resume at step '{todo[0][0]}' - {n} files of the finished steps fetched from S3")
    for n, c, outs in todo:
        t1 = time.time()
        run(c.format(S=root, name=name), env, lf)
        S3.Stage(prefix, f"step_{n}").mark(S3.upload_dir(root, prefix, _expand(root, outs)), t1)
        log(f"process: step {n} done in {time.time() - t1:.0f} s (on S3)")
    upload_logs(root, prefix)


def site_env(p: Params, site: Site, root: Path) -> dict:
    env = dict(os.environ)
    env.update({k: str(v) for k, v in p.extra["download_env"].items()})
    env.update(site.track_env())
    env.update({"SC_ROOT": str(root), "SC_AOI": str(root / "data/aoi/aoi.shp"), "GDAL_NUM_THREADS": "ALL_CPUS",
                "DL_START": p.season_start, "DL_END": p.season_end, "PYTHONUNBUFFERED": "1",
                "DL_WORKLOAD_TAG": f"{p.workload_tag}-{p.province.lower().replace('_', '-')}"})
    if site.kind == "gap":
        env.update({k: str(v) for k, v in p.extra["gap_download_env"].items()})
    return env


def stage_download(p: Params, site: Site, root: Path, env: dict, prefix: str):
    st = S3.Stage(prefix, "download")
    if st.done() and not p.forced("download"):
        if S3.Stage(prefix, "step_stacks").done() and not p.forced("process"):
            log("download: done; stacks already on S3 -> raw not needed on this pod"); return
        if not any((root / "data/raw").rglob("*.tif")):
            wait_for_disk(p)
            n = S3.download_prefix(prefix, root, only=UPLOAD_DOWNLOAD)
            log(f"download: done earlier -> {n} raw files fetched from S3")
        else:
            log("download: done (raw on disk)")
        return
    t0 = time.time()
    for only in ("s1", "s2"):
        for attempt in (1, 2, 3):
            wait_for_disk(p)
            lf = root / f"logs/download_{only}.log"
            try:
                run(f"python -u scripts/download.py --only {only}", env, lf)
                if " 0 failed" in lf.read_text().rsplit("### ", 1)[-1]:
                    break
            except RuntimeError as e:
                log(f"download {only} attempt {attempt}: {str(e).splitlines()[0]}")
        else:
            raise RuntimeError(f"download {only} of {site.name} still has failed files after 3 attempts (see {lf})")
    files = S3.upload_dir(root, prefix, UPLOAD_DOWNLOAD); upload_logs(root, prefix)
    st.mark(files, t0, params={"season": [p.season_start, p.season_end], "tracks": [site.des, site.asc]})
    log(f"download: done in {(time.time() - t0) / 60:.0f} min, {len(files)} files on S3")


def process_one(p: Params, root: Path, name: str, env: dict):
    lf = root / "logs/process.log"
    (root / DONE_LOCAL).unlink(missing_ok=True)
    for c in process_cmds(p):
        run(c.format(S=root, name=name), env, lf)
    (root / DONE_LOCAL).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))   # all steps finished (a killed run leaves no sentinel)


def stage_process(p: Params, site: Site, root: Path, env: dict, prefix: str):
    st = S3.Stage(prefix, "process")
    t0 = time.time()
    lf = root / "logs/process.log"
    if p.forced("process"):                     # redo: drop the outputs so every step runs again
        shutil.rmtree(root / "data/model", ignore_errors=True)
    tiles = []
    if site.kind != "gap":
        run_steps(p, root, site.name, env, prefix)
        files = S3.upload_dir(root, prefix, UPLOAD_PROCESS) if (root / "data/stacks").exists() else \
            {k: v for k, v in S3.listing(prefix).items() if k.startswith(tuple(UPLOAD_PROCESS))}
    else:
        if not (root / "data/stacks/stacks.csv").exists():
            run("python -u scripts/build_stacks.py", env, lf)                                       # gap: whole-band steps exceed the pod memory on big sparse grids -> tiles
        if not (root / "tiles.txt").exists():
            run(f"python -u scripts/split_site.py {site.name}", env, root / "logs/split.log")
        tiles = (root / "tiles.txt").read_text().split()
        for tn in tiles:
            T = SP.site_dir(tn)
            tprefix = site_prefix(p, tn)
            tst = S3.Stage(tprefix, "process")
            if tst.done() and not p.forced("process"):
                continue
            t1 = time.time()
            tenv = dict(env, SC_ROOT=str(T), SC_AOI=str(T / "data/aoi/aoi.shp"))
            if not (T / DONE_LOCAL).exists():
                process_one(p, T, tn, tenv)
            tst.mark(S3.upload_dir(T, tprefix, [s for s in UPLOAD_PROCESS if s != "data/stacks"]), t1, parent=site.name)
            log(f"process: tile {tn} done")
        files = S3.upload_dir(root, prefix, UPLOAD_PROCESS)
    st.mark(files, t0, tiles=tiles, models={"layer1": p.layer1_model, "colour": p.colour_model})
    log(f"process: done in {(time.time() - t0) / 60:.0f} min, {len(files)} files on S3")
    upload_logs(root, prefix)
    if not p.keep_local:                        # S3 is the storage: the pod keeps nothing of a finished site but its logs
        shutil.rmtree(root / "data", ignore_errors=True)
        for tn in tiles:
            shutil.rmtree(SP.site_dir(tn) / "data", ignore_errors=True)


def upload_logs(root: Path, prefix: str):
    """The site's logs next to its data on S3 (<prefix>/logs/), so a run on another pod can be read later."""
    try:
        S3.upload_dir(root, prefix, ["logs"])
    except Exception as e:                      # a log upload never fails the stage
        log(f"log upload skipped: {str(e)[:120]}")


def run_site(p: Params, site: Site):
    root = site_root(p, site.name)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    prefix = site_prefix(p, site.name)
    if S3.Stage(prefix, "process").done() and not (p.forced("process") or p.forced("download")):
        log(f"{site.name}: done on S3 - skipped"); return
    t0 = time.time()
    log(f"{site.name} ({p.province}, {site.kind}, DES RO{site.des:03d} ASC RO{site.asc:03d}) -> {prefix}")
    ensure_shared(p)
    ensure_aoi(p, site, root)
    env = site_env(p, site, root)
    stage_download(p, site, root, env, prefix)
    stage_process(p, site, root, env, prefix)
    log(f"{site.name}: DONE in {(time.time() - t0) / 60:.0f} min")

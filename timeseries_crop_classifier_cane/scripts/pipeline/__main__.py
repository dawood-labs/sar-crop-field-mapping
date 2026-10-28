"""Standing-cane pipeline: give a province, the rest is automatic and resumable from S3.

  python -m pipeline run --province province5 [--units grid --grid-km 20] [--fd true --fd-versions v1,2ac]
        = prepare -> every site (download -> process) -> province outputs -> field delineation (when --fd true)
Single tasks (one per Airflow task; each skips what S3 already holds):
  prepare  --province P            units (districts + gaps, or grid tiles), S1 tracks, AOIs -> S3 plan
  plan     --province P            JSON task list for a DAG (site tasks, province task, fd tasks, done flags)
  site     --province P --site S   one unit: download -> process -> S3
  province --province P            mosaic, clean, clip, polygons -> S3
  fd       --province P --version V  one field-delineation version -> S3
  status   [--province P]          what is done on S3
  backfill --province P            markers for work done before this runner existed (JupyterHub pod)
Every parameter (see config.py Params) can also come from env PIPELINE_<NAME> or config/standing_cane_2026.json.
Run from the project root:  PYTHONPATH=scripts python -m pipeline <command> ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields

from . import s3state as S3
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import site_paths as SP  # noqa: E402
from .config import PROJ, Params, load_params, load_plan
from .province import fd, outputs
from .site import log, run_site, site_prefix


def plan(p: Params) -> dict:
    prov = load_plan(p); pre = p.prefix()
    sites = [{"task": f"site:{s.name}", "site": s.name, "kind": s.kind,
              "done": S3.Stage(site_prefix(p, s.name), "process").done()} for s in prov.sites]
    return {"province": p.province, "units": prov.units, "sites": sites,
            "province_outputs": {"task": "province", "after": "all sites", "done": S3.Stage(f"{pre}/province", "outputs").done()},
            "fd": [{"task": f"fd:{v}", "after": "province", "done": S3.Stage(f"{pre}/fd_review", f"fd_{v}").done()}
                   for v in (p.fd_list if p.fd else [])]}


def status(p: Params, provs: list[str]):
    for prov in provs:
        p.province = prov
        try:
            pl = plan(p)
        except RuntimeError as e:
            print(f"{prov}: {e}"); continue
        todo = [s["site"] for s in pl["sites"] if not s["done"]]
        fdl = [(v, S3.Stage(f"{p.prefix()}/fd_review", f"fd_{v}").done()) for v in p.fd_list]
        print(f"{prov} ({pl['units']}): sites {len(pl['sites']) - len(todo)}/{len(pl['sites'])}; province outputs "
              f"{'done' if pl['province_outputs']['done'] else 'todo'}; fd " + " ".join(f"{v}={'done' if d else 'todo'}" for v, d in fdl))
        if todo:
            print("   todo:", ", ".join(todo))


def run_all(p: Params):
    from .prepare import prepare
    prov = prepare(p)
    if p.dry_run:
        print(json.dumps(prov.to_json(), indent=1)); return
    failed = []
    for s in prov.sites:
        try:
            run_site(p, s)
        except Exception as e:                       # one bad site must not stop the others; the province step waits for all
            failed.append(s.name); log(f"{s.name}: FAILED - {str(e).splitlines()[0]}")
    if failed:
        sys.exit(f"{p.province}: {len(failed)} sites failed ({', '.join(failed)}); run again to retry only those")
    outputs(p, prov)
    if p.fd:
        for i, v in enumerate(p.fd_list):
            fd(p, v, cleanup=i == len(p.fd_list) - 1)
    log(f"{p.province}: ALL DONE")


def backfill(p: Params):
    """Markers for sites / province outputs / fd versions finished by the earlier drivers (checked against the S3 listing)."""
    prov = load_plan(p); pre = p.prefix(); t0 = time.time()
    for s in prov.sites:
        sp = site_prefix(p, s.name)
        if S3.Stage(sp, "process").done():
            continue
        have = S3.listing(sp); tiles = []
        local = SP.SITES / p.year / p.province / s.name
        if s.kind == "gap":
            tf = local / "tiles.txt"
            tiles = tf.read_text().split() if tf.exists() else []
            ok = bool(tiles)
            for tn in tiles:
                tp = site_prefix(p, tn); th = S3.listing(tp)
                if "data/model/layer2_final/class3_2026.tif" not in th:
                    ok = False; break
                if not S3.Stage(tp, "process").done():
                    S3.Stage(tp, "process").mark({k: v for k, v in th.items() if k.startswith("data/")}, t0, parent=s.name, backfilled=True)
        else:
            ok = "data/model/layer2_final/class3_2026.tif" in have
        if not ok and s.kind == "district" and (local / "data/model/layer2_final/class3_2026.tif").exists():
            files = S3.upload_dir(local, sp, ["data/aoi", "data/grid", "data/manifest", "data/model", "data/features"])
            S3.Stage(sp, "process").mark(files, t0, tiles=[], backfilled=True, note="raw inputs only on GCS (old flat layout)")
            print(f"  {s.name}: local outputs uploaded + marked done ({len(files)} files)"); continue
        if not ok:
            print(f"  {s.name}: not finished on S3 - left for the runner"); continue
        raw = {k: v for k, v in have.items() if k.startswith("data/raw/")}
        if raw and not S3.Stage(sp, "download").done():
            S3.Stage(sp, "download").mark(raw, t0, backfilled=True)
        S3.Stage(sp, "process").mark({k: v for k, v in have.items() if k.startswith("data/") and not k.startswith("data/raw/")},
                                     t0, tiles=tiles, backfilled=True)
        print(f"  {s.name}: marked done ({len(have)} objects)")
    stem = p.province.lower(); have = S3.listing(f"{pre}/province")
    if f"{stem}_sugarcane_2026.gpkg" in have and not S3.Stage(f"{pre}/province", "outputs").done():
        S3.Stage(f"{pre}/province", "outputs").mark({k: v for k, v in have.items() if not k.startswith("_state/")}, t0, backfilled=True)
        print("  province outputs: marked done")
    O = SP.SITES / p.year / p.province / "fd_review"
    for v in p.fd_list:
        st = S3.Stage(f"{pre}/fd_review", f"fd_{v}"); sv = f"{stem}_fd_sugarcane_2026_{v}"
        if st.done() or not (O / f"{sv}_log.json").exists():
            continue
        keep = [x.name for x in O.glob(f"{sv}*") if x.is_file() and "precleaned" not in x.name and "checkpoint" not in x.name]
        st.mark(S3.upload_dir(O, f"{pre}/fd_review", keep), t0, backfilled=True)
        print(f"  fd {v}: uploaded + marked done")


def main():
    ap = argparse.ArgumentParser(prog="pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "prepare", "plan", "site", "province", "fd", "status", "backfill"])
    ap.add_argument("--site", help="site: the unit name (from `plan`)")
    ap.add_argument("--version", help="fd: v1 | 1ac | 1p5ac | 2ac")
    ap.add_argument("--local", action="store_true", help="prepare: plan from this pod's existing site folders (JupyterHub only)")
    for f in fields(Params):                          # every run parameter as a flag: --grid-km 20, --fd true, ...
        if f.name != "extra":
            ap.add_argument("--" + f.name.replace("_", "-"), dest=f.name, default=None)
    a = ap.parse_args()
    p = load_params({f.name: getattr(a, f.name) for f in fields(Params) if f.name != "extra"})
    if a.cmd != "status" and not p.province:
        ap.error("--province is required")
    if a.cmd == "run":
        run_all(p)
    elif a.cmd == "prepare":
        from .prepare import prepare
        prepare(p, local=a.local)
    elif a.cmd == "plan":
        print(json.dumps(plan(p), indent=1))
    elif a.cmd == "site":
        run_site(p, load_plan(p).site(a.site))
    elif a.cmd == "province":
        outputs(p, load_plan(p))
    elif a.cmd == "fd":
        fd(p, a.version)
    elif a.cmd == "status":
        status(p, [p.province] if p.province else sorted(x.stem for x in (PROJ / "config/provinces").glob("*.json")))
    elif a.cmd == "backfill":
        backfill(p)


if __name__ == "__main__":
    main()

"""Watchdog for the running pipeline jobs (user, 28 Oct 2026: "har process aur time par nazar rakho, koi redundancy nahi,
koi bug nahi"). Every INTERVAL s it checks and prints one ALERT line per problem (nothing when all is fine):
  stall      a job's log has not changed for STALL_MIN minutes while the job is alive
  died       a watched log ends with a failure (Traceback / FAILED / failed) and its job is gone
  duplicate  two processes work on the same site / output at the same time (wasted work, clobbered files)
  idle       load < 25 % of the cores for 2 checks while jobs are running (cores sitting idle)
  disk       free disk < 25 GB
  creds      AWS credentials expired or expiring (STS call fails)
usage: python scripts/watchdog.py [--once]      log: logs/watchdog.log (a Monitor on it shows the alerts)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
INTERVAL, STALL_MIN = int(os.environ.get("WD_INTERVAL", 300)), int(os.environ.get("WD_STALL_MIN", 25))
JOB = re.compile(r"pipeline (run|site|province|fd)|fd_review_province\.sh|fd_cane_fields\.py|scripts/download\.py|build_stacks\.py|"
                 r"s1_composites|s2_dry_features|pheno_features|apply_site|colour_multi|split_site|province_cane|province_mosaic")


def procs():
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
            env = dict(x.split("=", 1) for x in (d / "environ").read_bytes().decode(errors="ignore").split("\0") if "=" in x)
        except OSError:
            continue
        if JOB.search(cmd) and not cmd.startswith(("/bin/sh -c", "sh -c")) and "watchdog" not in cmd:
            out.append((int(d.name), cmd, env.get("SC_ROOT", "")))
    return out


def watched_logs():
    logs = list((PROJ / "logs").glob("runner_*.log")) + list((PROJ / "logs").glob("fd_review_*.log"))
    return [p for p in logs if time.time() - p.stat().st_mtime < 6 * 3600]


def check(state: dict) -> list[str]:
    al, ps = [], procs()
    alive = bool(ps)
    # duplicates: the same science step twice on one site, or two runners / fd jobs on one province
    seen = {}
    for pid, cmd, root in ps:
        key = None
        m = re.search(r"pipeline (run|site|province|fd) .*--province (\S+)", cmd)
        if m: key = ("runner", m.group(2))
        elif "fd_review_province.sh" in cmd: key = ("fdloop", cmd.split()[-1])
        elif "fd_cane_fields.py" in cmd: key = ("fd", cmd.split()[-1])
        elif root and "python" in cmd: key = ("step", root, re.sub(r".*scripts/(\w+\.py).*", r"\1", cmd))
        if key:
            seen.setdefault(key, []).append(pid)
    for k, pids in seen.items():
        # a bash loop forks subshells and multiprocessing pools fork workers: count only unrelated processes (different parents)
        tops = [p for p in pids if _ppid(p) not in pids]
        if len(tops) > 1:
            al.append(f"duplicate {k}: pids {tops}")
    # stalls / deaths per watched log
    for lg in watched_logs():
        age = (time.time() - lg.stat().st_mtime) / 60
        tail = lg.read_text(errors="ignore")[-3000:]
        last = tail.strip().splitlines()[-1] if tail.strip() else ""
        finished = bool(re.search(r"ALL DONE|FD_REVIEW_DONE", last))
        failed = bool(re.search(r"Traceback|FAILED|failed|Error", last))
        prov = lg.stem.split("_", 1)[1]
        running = any(prov.lower() in c.lower() or prov.replace("_", " ").lower() in c.lower() for _, c, _ in ps) or \
            any(prov.lower() in r.lower() for _, _, r in ps)
        if failed and not running:
            al.append(f"died {lg.name}: {last[:160]}")
        elif running and not finished and age > STALL_MIN and not _child_progress(prov):
            al.append(f"stall {lg.name}: no log change for {age:.0f} min (last: {last[:120]})")
    # idle cores while jobs run
    load = os.getloadavg()[0]; n = os.cpu_count() or 1
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        n = min(n, int(int(q) / int(p) + 0.5)) if q != "max" else n
    except Exception:
        pass
    state["idle"] = state.get("idle", 0) + 1 if alive and load < 0.25 * n else 0
    if state["idle"] >= 2:
        al.append(f"idle cores: load {load:.1f} on {n} cores with {len(ps)} job processes")
    free = shutil.disk_usage(PROJ).free / 2**30
    if free < 25:
        al.append(f"disk: only {free:.0f} GB free")
    if time.time() - state.get("creds_t", 0) > 1800:
        state["creds_t"] = time.time()
        env = dict(os.environ)
        f = os.environ.get("AWS_ENV", str(Path.home() / ".config/standing_cane/aws.env"))
        r = subprocess.run(f"source {f} 2>/dev/null; aws sts get-caller-identity --output text", shell=True, executable="/bin/bash",
                           capture_output=True, text=True, env=env, timeout=60)
        if r.returncode:
            al.append(f"creds: AWS STS call failed ({r.stderr.strip()[:120]}) -> uploads will fail; new credentials needed")
    return al


def _ppid(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except Exception:
        return -1


def _child_progress(prov: str) -> bool:
    """True when a site / fd log of this province changed recently (a long step logs there, not in the runner log)."""
    cut = time.time() - STALL_MIN * 60
    cands = list((PROJ / "sites/2026" / prov).glob("*/logs/*.log")) + list((PROJ / "sites/2026" / prov / "logs").glob("*.log")) + \
        list((PROJ / "logs").glob(f"fd_{prov.lower()}_*.log"))
    return any(p.stat().st_mtime > cut for p in cands)


def main():
    state = {}
    while True:
        for a in check(state):
            print(time.strftime("%H:%M"), "ALERT", a, flush=True)
        if "--once" in sys.argv:
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()

"""GEE export watchdog: every INTERVAL s, cancel any standing-cane export task that has been RUNNING for more than MAX_MIN minutes
(normal tasks finish in ~2 min; stuck ones sit at 29-57 % for hours and hold a whole district back), drop it from the district's
manifest and resubmit that one job with scripts/export_gcs.py (which skips everything already on GCS).
usage: setsid nohup python scripts/gee_watchdog.py > logs/gee_watchdog.log 2>&1 &     env MAX_MIN=45 INTERVAL=600"""
import os, re, subprocess, sys, time
import pandas as pd
EE_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1]); sys.path.insert(0, os.path.join(EE_ROOT, "scripts"))
import ee
from sar_pipeline.auth import credentials
import site_paths as SP
P = "."; MAX_MIN = int(os.environ.get("MAX_MIN", 45)); INTERVAL = int(os.environ.get("INTERVAL", 600))
ee.Initialize(credentials({"_root": EE_ROOT, "auth": {"key_file": __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json"), "project": __import__("os").environ.get("EE_PROJECT", "")}}), project=__import__("os").environ.get("EE_PROJECT", ""))
ENV = dict(os.environ, DL_S1_TRACKS="62:DESCENDING,172:ASCENDING", DL_START="2025-11-01", DL_END="2026-10-15", EE_MAX_ACTIVE="600")
while True:
    now = pd.Timestamp.utcnow().tz_localize(None); stuck = {}
    for o in ee.data.listOperations():
        md = o["metadata"]; d = md.get("description", "")
        if md.get("state") != "RUNNING" or not d.startswith("sc_") or "startTime" not in md: continue
        age = (now - pd.Timestamp(md["startTime"]).tz_localize(None)).total_seconds() / 60
        m = re.match(r"sc_(.+?_2026)_", d)
        if age > MAX_MIN and m: stuck.setdefault(m.group(1), []).append((o["name"].split("/")[-1], d, round(age), round(float(md.get("progress", 0) or 0), 2)))
    for site, tasks in stuck.items():
        f = f"{SP.site_dir(site)}/data/manifest/export_tasks.csv"
        if not os.path.exists(f): continue
        for tid, d, age, prog in tasks:
            try: ee.data.cancelOperation(f"projects/ee-company/operations/{tid}"); print(f"{now:%H:%M} cancelled {d} ({age} min, {prog})", flush=True)
            except Exception as e: print(f"{now:%H:%M} cancel failed {d}: {str(e)[:80]}", flush=True)
        m = pd.read_csv(f, dtype=str); m = m[~m.task_id.isin([t[0] for t in tasks])]; m.to_csv(f, index=False)
        S = f"{SP.site_dir(site)}"; env = dict(ENV, SC_ROOT=S, SC_AOI=f"{S}/data/aoi/aoi.shp")
        r = subprocess.run([sys.executable, f"{P}/scripts/export_gcs.py", "submit"], env=env, capture_output=True, text=True, cwd=P)
        print(f"{now:%H:%M} resubmitted {site}: " + " | ".join(l for l in r.stdout.splitlines() if "done:" in l or "to submit" in l), flush=True)
    if not stuck: print(f"{now:%H:%M} ok", flush=True)
    time.sleep(INTERVAL)

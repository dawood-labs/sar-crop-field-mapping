"""Make a site ready for manual review in QGIS / the notebook (user, 13 Oct 2026):
  - S2 chips with B2, B3, B4, B5, B8 (5-3-2 and 8-4-3)
  - EVERY October image of the season year (any cloud), and at least ONE image per month from March on: when the
    normal download (<= 20 % AOI cloud) has none for a month, the least cloudy date of that month is downloaded
  - a QGIS style (.qml, 5-3-2) next to every S2 file; S2 stacks rebuilt so the notebook sees the new dates
  - prints the clearest date per month and checks data/grid/index.tif (pixel ids) exists
usage: SC_ROOT=sites/<site> SC_AOI=<aoi.shp> python scripts/fill_months.py 2026
"""
from __future__ import annotations

import os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import site_paths as SP

PROJ = Path(__file__).resolve().parents[1]
ROOT = SP.site_dir(os.environ["SC_ROOT"])
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
BANDS = ("B2", "B3", "B4", "B5", "B8")
QML = (PROJ / "scripts/merge_s2_band.py").read_text().split('QML = """', 1)[1].split('"""', 1)[0]


def download(date):
    d = pd.Timestamp(date)
    env = dict(os.environ, DL_START=d.strftime("%Y-%m-%d"), DL_END=(d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
               DL_MAX_CLOUD="1.0", DL_S2_BANDS=",".join(BANDS), DL_S2_QA="", DL_WORKERS="32")
    subprocess.run([sys.executable, "-u", str(PROJ / "scripts/download.py"), "--only", "s2"], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    man_f = ROOT / f"data/manifest/s2_cloud_{YEAR - 1}_{YEAR}.csv"
    man = pd.read_csv(man_f, dtype={"date": str})          # every date of the season with its AOI cloud fraction
    keep = man.copy()
    have = {p.stem[3:] for p in (ROOT / "data/raw/s2").glob("S2_*.tif")}
    want = set(man[man.date.str[:6] == f"{YEAR}10"].date)                       # all of October
    for m in range(3, 11):
        mm = man[man.date.str[:6] == f"{YEAR}{m:02d}"]
        if len(mm) and not any(d[:6] == f"{YEAR}{m:02d}" for d in have):
            want.add(mm.sort_values("aoi_cloud_frac").date.iloc[0])               # least cloudy of that month
    todo = sorted(want - have)
    print("downloading:", todo, flush=True)
    from concurrent.futures import ThreadPoolExecutor        # dates in parallel (network-bound)
    with ThreadPoolExecutor(min(6, max(1, len(todo)))) as ex:
        list(ex.map(download, todo))
    keep.to_csv(man_f, index=False)                                               # the one-day runs overwrite it
    bad = []
    for f in sorted((ROOT / "data/raw/s2").glob("S2_*.tif")):
        if tuple(rasterio.open(f).descriptions) != BANDS:
            bad.append(f.name)
        f.with_suffix(".qml").write_text(QML)
    if not os.environ.get("FILL_NO_STACKS"):                       # batch runner builds stacks later, once
        subprocess.run([sys.executable, "-u", str(PROJ / "scripts/build_stacks.py")], env=dict(os.environ), check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    grid = ROOT / "data/grid/index.tif"
    ins = rasterio.open(grid).read(2, out_shape=(800, 800)).astype(bool)
    rows = []
    for f in sorted((ROOT / "data/raw/s2_cloud").glob("S2CLOUD_*.tif")):
        d = f.stem[8:]
        if d[:4] == str(YEAR) and int(d[4:6]) >= 3:
            cs = rasterio.open(f).read(2, out_shape=(800, 800))
            rows.append((d, round(100 * float(((cs >= 60) & (cs <= 100))[ins].mean()), 1)))
    t = pd.DataFrame(rows, columns=["date", "aoi_clear_pct"])
    t["month"] = t.date.str[:6]
    best = t.sort_values("aoi_clear_pct", ascending=False).groupby("month").head(1).sort_values("date")
    best.to_csv(ROOT / "data/manifest/best_s2_per_month.csv", index=False)
    print("files with wrong bands:", bad)
    print("pixel index:", grid, grid.exists())
    print("clearest per month:\n" + best.to_string(index=False))
    print("October:", t[t.month == f"{YEAR}10"].to_string(index=False))


if __name__ == "__main__":
    main()

"""One table + one text file listing, per 2026 district, everything needed for a manual review in QGIS / the notebook:
latest model maps, change 2025 -> 2026, S2 images (clearest per month from March + all October), pixel index raster.
usage: python scripts/review_index.py [model tag, default loop3]  -> sites/REVIEW_INDEX_2026.csv + .md
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import site_paths as SP

PROJ = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "loop3"
SITES = ["site37_2026", "site32_2026", "site44_2026", "site11_2026", "site42_2026",
         "site24_2026", "site23_2026", "site05_2026", "site38_2026", "site25_2026"]


def s2_dates(root):
    grid = root / "data/grid/index.tif"
    if not grid.exists():
        grid = PROJ / "data/grid/index.tif"
    ins = rasterio.open(grid).read(2, out_shape=(600, 600)).astype(bool)
    rows = []
    for f in sorted((root / "data/raw/s2_cloud").glob("S2CLOUD_2026*.tif")):
        d = f.stem[8:]
        if int(d[4:6]) >= 3 and (root / f"data/raw/s2/S2_{d}.tif").exists():
            cs = rasterio.open(f).read(2, out_shape=(600, 600))
            rows.append((d, round(100 * float(((cs >= 60) & (cs <= 100))[ins].mean()))))
    t = pd.DataFrame(rows, columns=["date", "clear"])
    t["m"] = t.date.str[:6]
    best = t.sort_values("clear", ascending=False).groupby("m").head(1).sort_values("date")
    octo = t[t.m == "202610"].sort_values("date")
    return (" ".join(f"{r.date}({r.clear}%)" for r in best.itertuples()),
            " ".join(f"{r.date}({r.clear}%)" for r in octo.itertuples()))


def main():
    out, md = [], [f"# 2026 districts: review files (model {TAG})\n",
                   "Paths relative to . Every raster has a .qml style next to it (QGIS loads it).",
                   "S2 files: bands 1-5 = B2, B3, B4, B5, B8 (style = 5-3-2; 8-4-3 = bands 5/3/2). Pixel id = band 1 of the index raster.",
                   "change_2025_2026: 0 non-cane both, 1 cane both, 2 cane 2025 only (lost), 3 cane 2026 only (new).\n"]
    for s in SITES:
        root = SP.site_dir(s)
        mdir = root / "data/model" / TAG
        idx = root / "data/grid/index.tif"
        if not idx.exists():
            idx = PROJ / "data/grid/index.tif"                     # first AOI: same grid in both years
        ch = mdir / "change_2025_2026.json"
        if ch.exists():
            c = json.loads(ch.read_text())
            a25, a26 = c["cane_2025_map_ac"], c["cane_2026_model_ac"]
            ref = "2025 = the client 2025 map"
        else:                                                      # first AOI: model 2025 vs model 2026
            sm = mdir / "summary.json"
            if not sm.exists():                                    # post-rule tag (e.g. *_oct): 2025 from the source model
                o = json.loads((mdir / "oct_rule.json").read_text()); sm = mdir.parent / o["src"] / "summary.json"
            c = json.loads(sm.read_text())
            a25, a26 = c["cane_acres"]["2025_model"], c["cane_acres"]["2026_model"]
            if (mdir / "oct_rule.json").exists():
                a26 = json.loads((mdir / "oct_rule.json").read_text())["cane_ac_after"]
            ref = "2025 = same model on 2025 data"
        best, octo = s2_dates(root)
        rel = lambda p: str(p.relative_to(PROJ))  # noqa: E731
        row = {"district": s.replace("_2026", ""), "cane_2025_ac": a25, "cane_2026_ac": a26,
               "change_pct": round(100 * (a26 - a25) / max(a25, 1), 1), "change_reference": ref,
               "map_2026": rel(mdir / "cane_map_2026.tif"), "prob_2026": rel(mdir / "cane_prob_2026.tif"),
               "change": rel(mdir / "change_2025_2026.tif"), "s2_folder": rel(root / "data/raw/s2"),
               "index_raster": rel(idx), "notebook_SITE": "site37" if s == "site37_2026" else s,
               "s2_clearest_per_month": best, "s2_october_all": octo}
        out.append(row)
        md += [f"## {row['district']}  (cane 2025 {a25:,} ac -> 2026 {a26:,} ac, {row['change_pct']:+.1f} %; {ref})",
               f"- map: `{row['map_2026']}`", f"- probability: `{row['prob_2026']}`", f"- change: `{row['change']}`",
               f"- S2 images: `{row['s2_folder']}/S2_YYYYMMDD.tif`", f"  - clearest per month: {best}",
               f"  - all October: {octo}", f"- pixel index: `{row['index_raster']}`",
               f"- notebook: `SITE = \"{row['notebook_SITE']}\"`, `SEASON = 2026`\n"]
    pd.DataFrame(out).to_csv(PROJ / "sites/REVIEW_INDEX_2026.csv", index=False)
    (PROJ / "sites/REVIEW_INDEX_2026.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

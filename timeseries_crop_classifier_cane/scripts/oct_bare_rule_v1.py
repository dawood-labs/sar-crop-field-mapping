"""Post-rule (independent review 14 Oct 2026): a pixel mapped cane is set to non-cane when ANY clear S2 date from
15 Sep 2026 shows NDVI < THR (0.30): the field was harvested / cleared, so nothing stands there in October.
On the review (133 mapped-cane blocks) this flagged 13 of 22 false cane and 0 of 111 true cane.
usage: python scripts/oct_bare_rule.py <src tag> <out tag> sites/<site> [...]  -> <site>/data/model/<out tag>/
         cane_prob_2026.tif (copy), cane_map_2026.tif (+ .qml), oct_ndvi_min.tif (x100, 255 = no clear date), oct_rule.json
"""
import json, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qgis_style as Q  # noqa: E402

THR, START = 0.30, pd.Timestamp("2026-09-15")


def one(site, src, dst):
    st, s, o = site / "data/stacks", site / "data/model" / src, site / "data/model" / dst
    o.mkdir(parents=True, exist_ok=True)
    with rasterio.open(s / "cane_map_2026.tif") as r:
        m, prof = r.read(1), r.profile.copy()
    dates = pd.to_datetime(list(rasterio.open(st / "S2_cs_cdf.tif").descriptions))
    idx = [i + 1 for i, d in enumerate(dates) if d >= START]
    mn = np.full(m.shape, np.inf, "float32")
    for i in idx:                                                   # one date at a time: low memory
        cs = rasterio.open(st / "S2_cs_cdf.tif").read(i)
        b8 = rasterio.open(st / "S2_B8.tif").read(i).astype("float32"); b4 = rasterio.open(st / "S2_B4.tif").read(i).astype("float32")
        ok = (cs >= 60) & (cs <= 100) & (b8 > 0)
        nd = (b8 - b4) / np.maximum(b8 + b4, 1)
        mn = np.where(ok, np.minimum(mn, nd), mn)
    bare = (m == 1) & (mn < THR)
    new = np.where(bare, 0, m).astype("uint8")
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    with rasterio.open(o / "cane_map_2026.tif", "w", **pr) as d:
        d.write(new, 1)
    with rasterio.open(o / "oct_ndvi_min.tif", "w", **pr) as d:
        d.write(np.where(np.isfinite(mn), np.clip(np.round(mn * 100), 0, 100), 255).astype("uint8"), 1)
    shutil.copy(s / "cane_prob_2026.tif", o / "cane_prob_2026.tif")
    for f in ("cane_prob_2026.qml",):
        if (s / f).exists():
            shutil.copy(s / f, o / f)
    Q.paletted(o / "cane_map_2026.tif", Q.MAP_CLASSES); Q.pseudocolor(o / "oct_ndvi_min.tif", Q.PROB_STOPS)
    ac = 100 / 4046.8564224
    res = {"site": site.name, "src": src, "thr": THR, "clear_dates_from_15sep": [str(dates[i - 1].date()) for i in idx],
           "cane_ac_before": round(float((m == 1).sum() * ac)), "cane_ac_after": round(float((new == 1).sum() * ac)),
           "removed_ac": round(float(bare.sum() * ac)),
           "cane_px_without_clear_oct_date": round(float(((m == 1) & ~np.isfinite(mn)).sum() / max((m == 1).sum(), 1)), 3)}
    (o / "oct_rule.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(lambda p: one(Path(p).resolve(), src, dst), sys.argv[3:]):
            print(r["site"], r["cane_ac_before"], "->", r["cane_ac_after"], f"(-{r['removed_ac']}),",
                  "no clear Oct date:", r["cane_px_without_clear_oct_date"], flush=True)

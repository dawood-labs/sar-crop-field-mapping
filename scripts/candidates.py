"""Create new labelled sample points from the data we already have (user, 2026-10-09): rule-based CANDIDATES for
sugarcane, cassava and fallow, then blind sheets for analyst labelling (the analyst never sees the rule).

Per-pixel summaries (cloud-free S2 NDVI medians per period; S1 monthly VH from data/features/s1_monthly_w5_v2.tif,
mean of both tracks; rice probability from model v3):
  sugarcane : green Dec-Jan (NDVI >= 0.5), cut Jan-Apr (lowest cloud-free NDVI <= 0.35), REGROWS in the dry season
              (Apr-May NDVI >= 0.35 and above Feb-Mar), no monsoon water (Jun VH > -19 dB), rice prob < 30
  cassava   : 2025 crop (Sep-Nov NDVI >= 0.45), still green Dec-Jan (>= 0.4), harvested and NOT regrowing in the dry
              season (Feb-Mar and Apr-May NDVI <= 0.35), no monsoon water (Jun VH > -19), canopy again by Aug-Sep
              (VH >= -17), rice prob < 30
  fallow    : user definition = something grew in the previous season but the land is EMPTY in the current (2026 rice)
              season: Sep-Nov 2025 NDVI >= 0.45, and Jul, Aug and Sep 2026 VH all <= -18 dB (no crop canopy), and Sep-Oct
              2026 NDVI <= 0.35 where a cloud-free image exists, rice prob < 50
60 pixels per class, >= 300 m apart and >= 300 m from every GCP, training sample and earlier audit pixel.
Outputs: data/labels_v1/key/candidates_key.csv (hidden: rule class), points.csv, sheets/C###.png (same blind sheet as
the audits), RUBRIC.md is written separately.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
OUT = PROJ / "data/labels_v1"
N_PER_CLASS = 60


def period_medians():
    st = PROJ / "data/stacks"
    with rasterio.open(st / "S2_NDVI.tif") as r:
        dates = pd.to_datetime(list(r.descriptions))
    P = {"SepNov25": ("2025-09-01", "2025-11-30"), "DecJan": ("2025-12-01", "2026-01-31"), "FebMar": ("2026-02-01", "2026-03-31"),
         "AprMay": ("2026-04-01", "2026-05-31"), "SepOct26": ("2026-09-01", "2026-10-08"), "JanApr": ("2026-01-01", "2026-04-30")}
    out = {}
    for k, (a, b) in P.items():                      # one period at a time keeps memory low (pod request is 16 GB)
        idx = [i + 1 for i, d in enumerate(dates) if pd.Timestamp(a) <= d <= pd.Timestamp(b)]
        with rasterio.open(st / "S2_NDVI.tif") as r:
            nd = r.read(idx).astype("float32") / 1e4
        with rasterio.open(st / "S2_CLEAR.tif") as r:
            nd[r.read(idx) != 1] = np.nan
        with np.errstate(all="ignore"):
            out[k] = np.nanmin(nd, 0) if k == "JanApr" else np.nanmedian(nd, 0)
        del nd
    return out


def main():
    t0 = time.time()
    (OUT / "key").mkdir(parents=True, exist_ok=True)
    nd = period_medians()
    with rasterio.open(PROJ / "data/features/s1_monthly_w5_v2.tif") as r:
        names = list(r.descriptions)
        def vh(m):
            i = [k + 1 for k, n in enumerate(names) if n.endswith(f"_VH_{m}")]
            return np.nanmean(r.read(i), 0)
        vh_jun, vh_jul, vh_aug, vh_sep = vh("202606"), vh("202607"), vh("202608"), vh("202609")
    prob = rasterio.open(PROJ / "data/model/rice_v3/rice_prob.tif").read(1).astype("float32")
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool) & (prob != 255)
    print(f"summaries in {time.time() - t0:.0f} s", flush=True)
    with np.errstate(invalid="ignore"):
        rules = {
            "sugarcane": (nd["DecJan"] >= 0.5) & (nd["JanApr"] <= 0.35) & (nd["AprMay"] >= 0.35) & (nd["AprMay"] > nd["FebMar"] + 0.1)
                         & (vh_jun > -19) & (prob < 30),
            "cassava": (nd["SepNov25"] >= 0.45) & (nd["DecJan"] >= 0.4) & (nd["FebMar"] <= 0.35) & (nd["AprMay"] <= 0.35)
                       & (vh_jun > -19) & (np.fmax(vh_aug, vh_sep) >= -17) & (prob < 30),
            "fallow": (nd["SepNov25"] >= 0.45) & (vh_jul <= -18) & (vh_aug <= -18) & (vh_sep <= -18)
                      & ~(nd["SepOct26"] > 0.35) & (prob < 50),
        }
    # pixels to stay away from: GCPs, all training samples, all earlier audits
    import geopandas as gpd
    from scipy.spatial import cKDTree
    av = [pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")[["row", "col"]].to_numpy()]
    for f in (PROJ / "data/model").glob("rice_v*/samples.gpkg"):
        av.append(gpd.read_file(f)[["row", "col"]].to_numpy())
    for f in (PROJ / "data/model").glob("rice_v*/audit/key/audit_key.csv"):
        av.append(pd.read_csv(f)[["row", "col"]].to_numpy())
    tav = cKDTree(np.vstack(av))
    rng = np.random.default_rng(2026)
    rows, pts = [], []
    for cls, m in rules.items():
        rr, cc = np.nonzero(m & inside)
        print(f"  {cls}: {len(rr)} candidate px ({len(rr) * 100 / 4046.856:,.0f} acres)", flush=True)
        got = 0
        for i in rng.permutation(len(rr)):
            p = (rr[i], cc[i])
            if tav.query(p)[0] < 30 or (pts and cKDTree(pts).query(p)[0] < 30):
                continue
            pts.append(p); rows.append(dict(row=int(p[0]), col=int(p[1]), rule_class=cls, rice_prob_v3=int(prob[p]))); got += 1
            if got >= N_PER_CLASS:
                break
    d = pd.DataFrame(rows).sample(frac=1, random_state=11).reset_index(drop=True)
    d.insert(0, "audit_id", [f"C{i + 1:03d}" for i in range(len(d))])
    import explorer as ex
    from pyproj import Transformer
    W, H, tr, crs = ex._grid()
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    d["lon"], d["lat"] = t.transform((tr.c + (d.col + 0.5) * 10).values, (tr.f - (d.row + 0.5) * 10).values)
    d.to_csv(OUT / "key/candidates_key.csv", index=False)
    d[["audit_id", "lon", "lat"]].to_csv(OUT / "points.csv", index=False)
    print(d.rule_class.value_counts().to_dict(), flush=True)

    # blind sheets: reuse the audit sheet drawer, writing into data/labels_v1/sheets
    sys.argv = [sys.argv[0], "rice_v3"]
    import audit_sheets as A
    A.A = OUT
    (OUT / "sheets").mkdir(exist_ok=True)
    T = A.templates()
    from concurrent.futures import ProcessPoolExecutor
    from safe_resources import workers
    n = workers(1.5 * 2**30)
    with ProcessPoolExecutor(n, initializer=_init) as pool:
        list(pool.map(_draw, [d.iloc[i::n] for i in range(n)], [T] * n))
    print(f"{len(list((OUT / 'sheets').glob('*.png')))} sheets in {time.time() - t0:.0f} s")


def _init():
    sys.argv = [sys.argv[0], "rice_v3"]
    import audit_sheets as A
    A.A = OUT


def _draw(chunk, T):
    import audit_sheets as A
    A.A = OUT
    A._draw(chunk, T)


if __name__ == "__main__":
    main()

"""Second, relaxed fallow candidate set (the strict rule in candidates.py found only 24 acres). Idle fields with weeds
reach VH -16..-18 dB, so: Sep-Nov 2025 NDVI >= 0.45 (something grew), Aug and Sep 2026 VH <= -16.5 dB and VH rose by
less than 2 dB from Jun to Sep (no crop canopy building), Sep-Oct 2026 NDVI <= 0.4 where cloud-free, not a tree
(lowest Jan-Apr NDVI <= 0.4), rice prob v3 < 50. 50 pixels, same spacing rules; appended to the same key as C130+.
"""
from __future__ import annotations
from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")
import sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import candidates as C


def main():
    nd = C.period_medians()
    with rasterio.open(PROJ / "data/features/s1_monthly_w5_v2.tif") as r:
        names = list(r.descriptions)
        vh = lambda m: np.nanmean(r.read([k + 1 for k, n in enumerate(names) if n.endswith(f"_VH_{m}")]), 0)
        jun, aug, sep = vh("202606"), vh("202608"), vh("202609")
    prob = rasterio.open(PROJ / "data/model/rice_v3/rice_prob.tif").read(1).astype("float32")
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool) & (prob != 255)
    with np.errstate(invalid="ignore"):
        m = ((nd["SepNov25"] >= 0.45) & (aug <= -16.5) & (sep <= -16.5) & (sep - jun < 2) & ~(nd["SepOct26"] > 0.4)
             & (nd["JanApr"] <= 0.4) & (prob < 50))
    rr, cc = np.nonzero(m & inside)
    print(f"relaxed fallow: {len(rr)} px ({len(rr) * 100 / 4046.856:,.0f} acres)", flush=True)
    import geopandas as gpd
    from scipy.spatial import cKDTree
    key = pd.read_csv(C.OUT / "key/candidates_key.csv")
    av = [pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")[["row", "col"]].to_numpy(), key[["row", "col"]].to_numpy()]
    av += [gpd.read_file(f)[["row", "col"]].to_numpy() for f in (PROJ / "data/model").glob("rice_v*/samples.gpkg")]
    av += [pd.read_csv(f)[["row", "col"]].to_numpy() for f in (PROJ / "data/model").glob("rice_v*/audit/key/audit_key.csv")]
    tav = cKDTree(np.vstack(av)); rng = np.random.default_rng(77); pts, rows = [], []
    for i in rng.permutation(len(rr)):
        p = (rr[i], cc[i])
        if tav.query(p)[0] < 30 or (pts and cKDTree(pts).query(p)[0] < 30):
            continue
        pts.append(p); rows.append(dict(row=int(p[0]), col=int(p[1]), rule_class="fallow_relaxed", rice_prob_v3=int(prob[p])))
        if len(pts) >= 50:
            break
    d = pd.DataFrame(rows)
    d.insert(0, "audit_id", [f"C{len(key) + i + 1:03d}" for i in range(len(d))])
    import explorer as ex
    from pyproj import Transformer
    W, H, tr, crs = ex._grid(); t = Transformer.from_crs(crs, 4326, always_xy=True)
    d["lon"], d["lat"] = t.transform((tr.c + (d.col + 0.5) * 10).values, (tr.f - (d.row + 0.5) * 10).values)
    pd.concat([key, d]).to_csv(C.OUT / "key/candidates_key.csv", index=False)
    pd.concat([key, d])[["audit_id", "lon", "lat"]].to_csv(C.OUT / "points.csv", index=False)
    sys.argv = [sys.argv[0], "rice_v3"]
    import audit_sheets as A
    A.A = C.OUT
    T = A.templates()
    from concurrent.futures import ProcessPoolExecutor
    from safe_resources import workers
    n = workers(1.5 * 2**30)
    with ProcessPoolExecutor(n, initializer=C._init) as pool:
        list(pool.map(C._draw, [d.iloc[i::n] for i in range(n)], [T] * n))
    print(len(d), "fallow sheets added; total", len(list((C.OUT / "sheets").glob("*.png"))))


if __name__ == "__main__":
    main()

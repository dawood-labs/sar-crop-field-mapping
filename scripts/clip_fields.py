"""Clip one field map's class polygons with another map's class polygons (user, 2026-10-09: take the cane out of rice).

usage: clip_fields.py <target.parquet> <target class> <clipper.parquet> <clipper class> <out.parquet> [gap_m]
  e.g. clip_fields.py data/model/rice_v5/fields/fields_rice_v5.parquet Rice \
                      data/model/cane_v3/fields/fields_cane_v3.parquet Cane-Cassava \
                      data/model/rice_v5/fields/fields_rice_v5_clip_cane_v3.parquet
Only target polygons of <target class> are cut, and the output holds ONLY <target class> polygons (user, 2026-10-09). After the cut a polygon keeps
  - if it lost < 1 % of its area (edge contact, not a real overlap): the overlap is trimmed, all parts >= 1 m2 kept,
  - otherwise only the parts >= 0.15 acre (small leftovers and edge slivers are deleted).
  Result: no target-class polygon overlaps a clipper-class polygon. With gap_m > 0 (e.g. 0.1, as inside each layer)
  the clipper is buffered by gap_m first, so the two layers do not even touch: a dissolve of both together never
  merges a target polygon with a clipper polygon.
Writes <out>.parquet (+ .shp next to it) with area_acres recomputed and columns clip_lost_acres, clip_status
(unchanged | trimmed | clipped | removed), plus <out>_summary.json. The clipper file is never changed.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, sys, time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
import topology as tp  # noqa: E402

ACRE_M2 = 4046.856
MIN_PART_AC, KEEP_IF_LOST = 0.15, 0.01


def main():
    import geopandas as gpd, pandas as pd, shapely
    t0 = time.time()
    tgt_p, tgt_cls, clp_p, clp_cls, out_p = sys.argv[1:6]
    gap = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    out_p = Path(out_p)
    T = gpd.read_parquet(tgt_p); C = gpd.read_parquet(clp_p)
    C = C[C["class"] == clp_cls].reset_index(drop=True)
    sel = np.nonzero((T["class"] == tgt_cls).values)[0]
    G = np.asarray(T.geometry.values[sel], dtype=object); CG = np.asarray(C.geometry.values, dtype=object)
    # target x clipper pairs that really overlap (bbox candidates, exact test in threads)
    tree = shapely.STRtree(CG)
    q = tree.query(G)
    ov = tp._threaded(lambda c: shapely.area(shapely.intersection(G[c[:, 0]], CG[c[:, 1]])), q.T.copy()) if q.shape[1] else np.zeros(0)
    if gap > 0:     # every pair closer than the gap is cut, touching ones included
        b = shapely.bounds(G)
        q = tree.query(shapely.box(b[:, 0] - gap, b[:, 1] - gap, b[:, 2] + gap, b[:, 3] + gap))
        q = q[:, tp._threaded(lambda c: shapely.dwithin(G[c[:, 0]], CG[c[:, 1]], gap), q.T.copy()).astype(bool)] if q.shape[1] else q
        CG_cut = tp._threaded(lambda g: shapely.buffer(g, gap, quad_segs=2), CG)
    else:
        q = q[:, ov > 0.01]
        CG_cut = CG
    print(f"{len(sel):,} {tgt_cls} polygons, {len(CG):,} {clp_cls} polygons; overlapping pairs {q.shape[1]:,} "
          f"({ov.sum() / ACRE_M2:,.0f} ac) ({time.time() - t0:.0f} s)", flush=True)
    A = shapely.area(G)
    status = np.array(["unchanged"] * len(G), dtype=object); lost = np.zeros(len(G))
    if q.shape[1]:
        q = q[:, np.argsort(q[0], kind="stable")]
        hit, first, cnt = np.unique(q[0], return_index=True, return_counts=True)
        cut = np.empty(len(hit), dtype=object)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(tp._workers(0.3 * 2**30)) as pool:
            cut[:] = list(pool.map(lambda k: shapely.union_all(CG_cut[q[1][first[k]:first[k] + cnt[k]]]), range(len(hit))))
        pair = np.empty((len(hit), 2), dtype=object); pair[:, 0] = G[hit]; pair[:, 1] = cut
        D = tp._threaded(lambda c: shapely.make_valid(shapely.difference(c[:, 0], c[:, 1])), pair)
        for k, i in enumerate(hit):
            frac = 1 - shapely.area(D[k]) / A[i]
            small = frac < KEEP_IF_LOST                   # edge contact only: trim the overlap, keep every real part
            parts = [p for p in shapely.get_parts(D[k]) if p.geom_type == "Polygon"
                     and (p.area >= 1.0 if small else p.area / ACRE_M2 >= MIN_PART_AC)]
            G[i] = shapely.multipolygons(parts) if len(parts) > 1 else (parts[0] if parts else shapely.Polygon())
            status[i] = "trimmed" if small else ("clipped" if parts else "removed")
            lost[i] = (A[i] - shapely.area(G[i])) / ACRE_M2
    T = T.copy()
    T["clip_status"] = "not_target"; T["clip_lost_acres"] = 0.0
    T.loc[T.index[sel], "clip_status"] = status; T.loc[T.index[sel], "clip_lost_acres"] = lost.round(4)
    T.loc[T.index[sel], "geometry"] = G
    T = T[~T.is_empty & (T["class"] == tgt_cls)].reset_index(drop=True)     # output: target class only (user)
    T["area_acres"] = (T.area / ACRE_M2).round(4)
    T.to_parquet(out_p, compression="zstd")
    shp = T.rename(columns={c: c[:10] for c in T.columns if c != "geometry"})
    shp.rename(columns={"clip_statu": "clip_stat", "clip_lost_": "clip_lost"}).to_file(out_p.with_suffix(".shp"))
    tgt = T
    pr = shapely.STRtree(CG).query(np.asarray(tgt.geometry.values, dtype=object), predicate="intersects")
    ov_after = float(shapely.area(shapely.intersection(np.asarray(tgt.geometry.values, dtype=object)[pr[0]], CG[pr[1]])).sum()) if pr.shape[1] else 0.0
    n_int = int(pr.shape[1])
    tg = np.asarray(tgt.geometry.values, dtype=object)
    pq = shapely.STRtree(CG).query(tg, predicate="dwithin", distance=1.0)
    min_gap = float(shapely.distance(tg[pq[0]], CG[pq[1]]).min()) if pq.shape[1] else float("nan")
    s = {"target": tgt_p, "clipper": clp_p, f"{tgt_cls}_acres_before": float(A.sum() / ACRE_M2),
         f"{tgt_cls}_acres_after": float(tgt.area_acres.sum()), "acres_removed": float(lost.sum()),
         f"{tgt_cls}_polygons_before": int(len(sel)), f"{tgt_cls}_polygons_after": int(len(tgt)),
         "status": pd.Series(status).value_counts().to_dict(), "overlap_acres_before": float(ov.sum() / ACRE_M2),
         f"touching_pairs_{tgt_cls}_vs_{clp_cls}_after": n_int, "overlap_m2_after": ov_after, "min_gap_m_after": min_gap, "gap_m": gap,
         "params": {"MIN_PART_AC": MIN_PART_AC, "KEEP_IF_LOST": KEEP_IF_LOST}}
    Path(str(out_p).replace(".parquet", "_summary.json")).write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(s, indent=1, default=lambda v: round(float(v), 2)))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

"""Field-level rice map from rice v5 + the field delineation (user, 2026-10-09).

1. Sieve: data/model/rice_v5/rice_map.tif, patches (rice or non-rice) smaller than 0.2 acre (<= 8 px, 8-connected)
   are merged into their neighbour -> fields/rice_map_sieve.tif.
2. Each delineation polygon (data/fields/delineation/..._fields_clean.parquet, clipped to the AOI) gets the majority class
   of the sieved pixels whose centre lies inside it (no pixel centre: the pixel under its centroid).
3. A BIG polygon (>= 1 acre) whose minority class is >= 25 % is not labelled whole: it is cut with STRAIGHT lines
   (never along the pixel staircase). Candidate directions = the field's own axes (minimum rotated rectangle) and the
   best linear rice / non-rice separator (logistic regression on pixel centres); the cut position maximises pixel
   agreement, and a cut is kept only if the two sides get different majority classes. Pieces are cut again while
   they stay big and mixed (max depth 4).
   A piece smaller than 0.15 acre on the SIDE of the parent field (>= 40 % of its outline on the parent's outline) is
   deleted; if it lies inside the parent (less on the outline), that cut is not made.
   The delineation is the cleaned file from scripts/simplify_fields.py (0.5 m simplified, nested merged, 0.1 m gaps);
   the output gets the same 0.1 m separation, so no two polygons overlap or touch (a dissolve never merges fields).
4. Rice outside every polygon: sieved rice pixels not in any polygon are polygonised, the existing polygons are
   subtracted, and pieces >= 0.2 acre are added as new rice polygons (source = gap_fill).
Outputs (data/model/rice_v5/fields/): fields_rice_v5.parquet (GeoParquet) + .shp, rice_map_sieve.tif, summary.json,
  deleted_side_slivers.parquet
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401

# usage: field_map_v5.py [model_dir_name] [class name] [map file]   (defaults: rice_v5 Rice rice_map.tif)
#        e.g. field_map_v5.py cane_v1 Cane-Cassava cane_map.tif  -> data/model/cane_v1/fields/fields_cane_v1.parquet
M = PROJ / "data/model" / (sys.argv[1] if len(sys.argv) > 1 else "rice_v5")
CLS = sys.argv[2] if len(sys.argv) > 2 else "Rice"
MAPF = sys.argv[3] if len(sys.argv) > 3 else "rice_map.tif"
NON, TAG, C = "Non-" + CLS.lower(), M.name, CLS.lower()[:4]          # C: short prefix for columns / keys (rice, suga)
C = {"sugarcane": "cane", "cane-cassava": "cane"}.get(CLS.lower(), C)
OUT = M / "fields"
FIELDS = PROJ / "data/fields/delineation/fields_clean.parquet"
ACRE_M2 = 4046.856
PX_AC = 100 / ACRE_M2
SIEVE_AC, BIG_AC, MINORITY, MIN_PIECE_AC, SIDE_SHARE, MAX_DEPTH, GAP_MIN_AC = 0.2, 1.0, 0.25, 0.15, 0.4, 4, 0.2


def best_cut(xy, y, P):
    """Best straight cut: returns (normal, threshold, accuracy) or None."""
    from sklearn.linear_model import LogisticRegression
    c = xy - xy.mean(0)
    dirs = []
    rect = np.asarray(P.minimum_rotated_rectangle.exterior.coords)
    for k in (0, 1):                                    # the field's two axes -> cut normals
        e = rect[k + 1] - rect[k]
        if np.hypot(*e) > 0:
            dirs.append(e / np.hypot(*e))
    if 0 < y.sum() < len(y):
        w = LogisticRegression(class_weight="balanced").fit(c / (c.std() + 1e-9), y).coef_[0]
        if np.hypot(*w) > 0:
            dirs.append(w / np.hypot(*w))
    best = None
    for n in dirs:
        p = c @ n
        o = np.argsort(p); ps, ys = p[o], y[o]
        cut = np.nonzero(np.diff(ps) > 1e-6)[0]           # split between distinct positions
        if not len(cut):
            continue
        nl = cut + 1; rl = np.cumsum(ys)[cut]; nr = len(y) - nl; rr = ys.sum() - rl
        acc = (np.maximum(rl, nl - rl) + np.maximum(rr, nr - rr)) / len(y)
        diff = ((rl / nl) > 0.5) != ((rr / nr) > 0.5)       # the two sides must get different classes
        acc = np.where(diff, acc, -1)
        i = int(np.argmax(acc))
        if acc[i] > 0 and (best is None or acc[i] > best[2]):
            best = (n, (ps[cut[i]] + ps[cut[i] + 1]) / 2 + 0.0, float(acc[i]), xy.mean(0))
    return best


def halfplane(n, t, origin, side, L=1e5):
    from shapely.geometry import Polygon
    p0 = origin + t * n; d = np.array([-n[1], n[0]])
    s = n * L * (1 if side > 0 else -1)
    return Polygon([p0 - d * L, p0 + d * L, p0 + d * L + s, p0 - d * L + s])


def split(P, xy, y, root, depth=0):
    """List of (geometry, n_px, n_rice, kind) pieces; kind = whole | split."""
    import shapely
    area = P.area / ACRE_M2
    frac = y.mean() if len(y) else 0.0
    if area < BIG_AC or min(frac, 1 - frac) < MINORITY or depth >= MAX_DEPTH or len(y) < 4:
        return [(P, len(y), int(y.sum()), "whole" if depth == 0 else "split")]
    b = best_cut(xy, y, P)
    if b is None:
        return [(P, len(y), int(y.sum()), "whole" if depth == 0 else "split")]
    n, t, _, origin = b
    out, keep_px = [], []
    proj = (xy - origin) @ n
    for side in (-1, 1):
        G = P.intersection(halfplane(n, t, origin, side))
        for part in getattr(G, "geoms", [G]):
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            m = shapely.contains_xy(part, xy[:, 0], xy[:, 1]) & ((proj > t) if side > 0 else (proj <= t))
            if part.area / ACRE_M2 < MIN_PIECE_AC:
                on_edge = part.boundary.intersection(root.boundary.buffer(0.5)).length / max(part.boundary.length, 1e-9)
                if on_edge >= SIDE_SHARE:
                    out.append(("deleted", part, m)); continue
                return [(P, len(y), int(y.sum()), "whole" if depth == 0 else "split")]   # inner sliver: do not cut
            out.append(("keep", part, m))
    pieces = []
    for kind, part, m in out:
        if kind == "deleted":
            pieces.append((part, int(m.sum()), int(y[m].sum()), "deleted"))
        else:
            pieces += split(part, xy[m], y[m], root, depth + 1)
    return pieces


def _split_chunk(jobs):
    import shapely
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)                              # one BLAS thread per process (sklearn), else 8 x 8 threads fight
    out = []
    for i, wkb, xy, y in jobs:
        P = shapely.from_wkb(wkb)
        out += [(i, shapely.to_wkb(g), n_, nr_, kind) for g, n_, nr_, kind in split(P, xy, y, P)]
    return out


def main():
    t0 = time.time()
    import geopandas as gpd
    import shapely
    from rasterio import features
    from scipy import ndimage
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(M / MAPF) as r:
        rice = r.read(1); prof = r.profile.copy(); tr = r.transform
    valid = rice != 255
    size = int(np.ceil(SIEVE_AC / PX_AC))                 # 0.2 ac = 8.1 px -> keep patches of >= 9 px
    sv = features.sieve(np.where(valid, rice, 0).astype("uint8"), size=size, mask=valid, connectivity=8)
    sv = np.where(valid, sv, 255).astype("uint8")
    with rasterio.open(OUT / f"{C}_map_sieve.tif", "w", **dict(prof, count=1, dtype="uint8", nodata=255)) as dst:
        dst.write(sv, 1); dst.set_band_description(1, f"1 {CLS}, 0 {NON} ({TAG}, sieve {SIEVE_AC} acre = {size} px, 8-connected)")
    print(f"sieve: {C} {(rice == 1).sum() * PX_AC:,.0f} -> {(sv == 1).sum() * PX_AC:,.0f} ac ({time.time() - t0:.0f} s)", flush=True)

    f = gpd.read_parquet(FIELDS, columns=["uid", "Shp_Label", "Confidence", "geometry"])
    aoi = gpd.read_file(PROJ / "data/aoi/aoi.shp").to_crs(f.crs).union_all()
    inside = f.within(aoi)
    f.loc[~inside, "geometry"] = f.loc[~inside].intersection(aoi)
    f = f[~f.is_empty & f.geom_type.isin(["Polygon", "MultiPolygon"])].explode(index_parts=False).reset_index(drop=True)
    f = f[f.area > 1].reset_index(drop=True)
    f["fid_px"] = np.arange(1, len(f) + 1)
    fid = features.rasterize(zip(f.geometry, f.fid_px), out_shape=sv.shape, transform=tr, fill=0, dtype="int32")
    m = (fid > 0) & valid
    npx = np.bincount(fid[m], minlength=len(f) + 1)[1:]
    nri = np.bincount(fid[m], weights=(sv[m] == 1), minlength=len(f) + 1)[1:].astype(int)
    cen = f.geometry.representative_point()
    rr, cc = rasterio.transform.rowcol(tr, cen.x.values, cen.y.values)
    rr, cc = np.clip(rr, 0, sv.shape[0] - 1), np.clip(cc, 0, sv.shape[1] - 1)
    f["n_px"], f["n_rice_px"] = npx, nri
    nopx = npx == 0
    f.loc[nopx, "n_rice_px"] = (sv[rr, cc] == 1)[nopx].astype(int); f.loc[nopx, "n_px"] = 0
    f["area_acres"] = f.area / ACRE_M2
    frac = np.where(npx > 0, nri / np.maximum(npx, 1), f.n_rice_px)
    cand = (f.area_acres >= BIG_AC) & (np.minimum(frac, 1 - frac) >= MINORITY) & (npx >= 4)
    print(f"polygons {len(f):,} in AOI; big & mixed (>= {BIG_AC} ac, minority >= {MINORITY:.0%}): {int(cand.sum()):,} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---- split the big mixed polygons
    rows = []
    ci = np.nonzero(cand.values)[0]
    pix_r, pix_c = np.nonzero(m)
    order = np.argsort(fid[pix_r, pix_c]); pr, pc = pix_r[order], pix_c[order]
    fsorted = fid[pr, pc]
    starts = np.searchsorted(fsorted, f.fid_px.values[ci]); ends = np.searchsorted(fsorted, f.fid_px.values[ci], side="right")
    jobs = []
    for k, i in enumerate(ci):
        r_, c_ = pr[starts[k]:ends[k]], pc[starts[k]:ends[k]]
        xy = np.c_[tr.c + (c_ + 0.5) * tr.a, tr.f + (r_ + 0.5) * tr.e]
        jobs.append((int(i), shapely.to_wkb(f.geometry.iat[i]), xy, (sv[r_, c_] == 1).astype(int)))
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from safe_resources import workers
    nw = workers(1.0 * 2**30)                         # all safe workers; each holds only its own polygons
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("fork")) as pool:
        res = [r for chunk in pool.map(_split_chunk, [jobs[k::nw * 4] for k in range(nw * 4)]) for r in chunk]
    for i, g, n_, nr_, kind in res:
        rows.append(dict(uid=f.uid.iat[i], Shp_Label=f.Shp_Label.iat[i], Confidence=f.Confidence.iat[i],
                         n_px=n_, n_rice_px=nr_, source="split" if kind != "whole" else "delineation",
                         deleted=kind == "deleted", geometry=shapely.from_wkb(g)))
    sp = gpd.GeoDataFrame(rows, crs=f.crs)
    n_split = int((sp.groupby("uid").size() > 1).sum()) if len(sp) else 0
    print(f"split: {n_split:,} of {len(ci):,} big mixed polygons cut into {int((~sp.deleted).sum()):,} pieces, "
          f"{int(sp.deleted.sum())} side slivers < {MIN_PIECE_AC} ac deleted ({time.time() - t0:.0f} s)", flush=True)
    keep = f[~cand].assign(source="delineation", deleted=False)[["uid", "Shp_Label", "Confidence", "n_px", "n_rice_px",
                                                                 "source", "deleted", "geometry"]]
    out = pd.concat([keep, sp], ignore_index=True)
    dele = out[out.deleted]; out = out[~out.deleted].drop(columns="deleted")

    # ---- rice outside every polygon -> new polygons
    gap = (sv == 1) & (fid == 0)
    shp = [shapely.geometry.shape(g) for g, v in features.shapes(gap.astype("uint8"), mask=gap, transform=tr, connectivity=8) if v == 1]
    G = gpd.GeoDataFrame(geometry=shp, crs=f.crs)
    G = G[G.area / ACRE_M2 >= GAP_MIN_AC * 0.5].reset_index(drop=True)          # pre-filter before the overlay
    tree = shapely.STRtree(out.geometry.values)
    def clean(g):
        idx = tree.query(g, predicate="intersects")
        return g.difference(shapely.union_all(out.geometry.values[idx])) if len(idx) else g
    G["geometry"] = [clean(g) for g in G.geometry]
    G = G.explode(index_parts=False).reset_index(drop=True)
    G = G[(G.geom_type == "Polygon") & (G.area / ACRE_M2 >= GAP_MIN_AC)].reset_index(drop=True)
    G = G.assign(uid=[f"gap_{i + 1}" for i in range(len(G))], Shp_Label=None, Confidence=np.nan,
                 n_px=(G.area / 100).round().astype(int), source="gap_fill")
    G["n_rice_px"] = G.n_px
    print(f"gap fill: {len(G):,} {C} polygons, {G.area.sum() / ACRE_M2:,.0f} ac ({time.time() - t0:.0f} s)", flush=True)
    out = pd.concat([out, G], ignore_index=True)
    # split pieces and gap-fill polygons touch each other: same 0.1 m separation as the cleaned delineation
    import topology as tp
    out["geometry"] = tp.separate(out.geometry.values)
    out = out[~out.is_empty].reset_index(drop=True)
    n_int, min_gap = tp.check(out.geometry.values)
    print(f"separated: intersecting pairs {n_int}, min gap {min_gap:.3f} m ({time.time() - t0:.0f} s)", flush=True)

    # ---- labels (majority rule) and attributes
    rf = np.where(out.n_px > 0, out.n_rice_px / out.n_px.clip(lower=1), out.n_rice_px.astype(float))
    out["rice_frac"] = np.round(rf, 3)
    out["class"] = np.where(rf > 0.5, CLS, NON)
    out["area_acres"] = (out.area / ACRE_M2).round(4)
    out["Confidence"] = out.Confidence.astype(float)
    out.insert(0, "poly_id", np.arange(1, len(out) + 1))
    out = gpd.GeoDataFrame(out, crs=f.crs)
    cols = ["poly_id", "uid", "source", "class", "rice_frac", "area_acres", "n_px", "n_rice_px", "Shp_Label", "Confidence", "geometry"]
    ren = {"rice_frac": f"{C}_frac", "n_rice_px": f"n_{C}_px"}
    out[cols].rename(columns=ren).to_parquet(OUT / f"fields_{TAG}.parquet", compression="zstd")
    shp_cols = {"poly_id": "poly_id", "uid": "uid", "source": "source", "class": "class", "rice_frac": f"{C}_frac",
                "area_acres": "area_ac", "n_px": "n_px", "n_rice_px": f"n_{C}_px", "Shp_Label": "shp_label", "Confidence": "fd_conf"}
    out[cols].rename(columns=shp_cols).to_file(OUT / f"fields_{TAG}.shp")
    if len(dele):
        dele.to_parquet(OUT / "deleted_side_slivers.parquet", compression="zstd")

    # ---- QA: polygon map vs the sieved raster
    lab = features.rasterize(zip(out.geometry, (out["class"] == CLS).astype(int) + 1), out_shape=sv.shape,
                             transform=tr, fill=0, dtype="uint8")
    ins = valid & rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    cov = ins & (lab > 0)
    agree = float(((lab[cov] == 2) == (sv[cov] == 1)).mean())
    s = {"sieve_px": size, "rice_acres_raster_v5": float((rice == 1).sum() * PX_AC), "rice_acres_sieved": float((sv == 1).sum() * PX_AC),
         "polygons": int(len(out)), "by_source": out.groupby("source").size().to_dict(),
         "rice_polygons": int((out["class"] == CLS).sum()), "rice_acres_polygons": float(out.loc[out["class"] == CLS, "area_acres"].sum()),
         "nonrice_acres_polygons": float(out.loc[out["class"] == NON, "area_acres"].sum()),
         "rice_acres_by_source": out[out["class"] == CLS].groupby("source").area_acres.sum().round(0).to_dict(),
         "big_mixed_polygons": int(len(ci)), "big_mixed_split": n_split, "deleted_side_slivers": int(len(dele)),
         "deleted_acres": float(dele.area.sum() / ACRE_M2) if len(dele) else 0.0,
         "pixel_agreement_polygons_vs_sieved": agree, "intersecting_pairs": n_int, "min_gap_m": min_gap, "aoi_px_covered_by_polygons": float(cov.sum() / ins.sum()),
         "params": dict(SIEVE_AC=SIEVE_AC, BIG_AC=BIG_AC, MINORITY=MINORITY, MIN_PIECE_AC=MIN_PIECE_AC, SIDE_SHARE=SIDE_SHARE,
                        MAX_DEPTH=MAX_DEPTH, GAP_MIN_AC=GAP_MIN_AC)}
    s = {k.replace("nonrice", f"non{C}").replace("rice", C): v for k, v in s.items()}
    s["class"] = CLS
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(s, indent=1, default=lambda v: round(float(v), 3)))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

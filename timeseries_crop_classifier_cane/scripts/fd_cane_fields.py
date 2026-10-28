"""Sugarcane FIELD polygons from the field delineation + the static (layer-2) class raster (user, 26 Oct 2026).
The delineation owns the geometry, the raster owns only the label. Only sugarcane polygons are kept and cleaned.

Rules (user, 26 Oct 2026):
- fields are cut at the AOI boundary (the delineation boundary gpkg); fields without any sugarcane are not touched
- sugarcane share = sugarcane px / (sugarcane + cassava + none) px, pixel centres (no all_touched); cloud / unknown px
  (65535 inside the AOI) are not counted (own column cloud_pct)
- share >= KEEP (0.75): the field is kept as it is (exact original geometry)
- share < KEEP but the non-cane part <= NONCANE_MAX_AC (1 ac, user 27 Oct: do not cut delineated fields): kept whole
  (review mixed_whole; non-cane = class-0 px only, cassava counts as cane-like); dropped only if cane < DROP_PCT (20) % AND < DROP_AC (0.2) acres. Fields must keep matching the basemap
  (analysts QC on Google basemap), so delineated outlines are cut only when really needed
- else (non-cane part > NONCANE_MAX_AC): straight cuts (along / across the long axis of the field's rotated rectangle, up to DEPTH cuts) and only
  the sugarcane piece is kept, if that piece has share >= KEEP, >= MIN_AC acres and is >= MIN_W m wide; else the field goes
- clean-up on the sugarcane polygons only: overlaps (smaller polygon wins), holes < HOLE_AC filled, tails/spikes
  (opening OPEN_M, mitre, clipped back to the outline; only applied when it removes >= 10 m2 and keeps > 50 % of the area),
  gaps <= 2 x GAP_M between neighbouring sugarcane polygons given to the neighbour with the longest shared edge,
  a SEP_M gap on each side between neighbours (no merge on dissolve), parts < MIN_AC dropped, area_ac in EPSG:32648
Cross-block duplicates of the delineation (the same field in two 10 km blocks, IoU >= 0.8) and nested copies (>= 90 % inside a
larger field) are removed first.
usage: python scripts/fd_cane_fields.py <fields.parquet> <aoi.gpkg> <class.tif> <out.gpkg> [--bbox xmin ymin xmax ymax]
"""
import argparse, json, os, time

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely import affinity
from shapely.geometry import box

KEEP = float(os.environ.get("KEEP", 0.75)); DEPTH = int(os.environ.get("DEPTH", 3))
MIN_AC = float(os.environ.get("MIN_AC", 0.15)); MIN_W = float(os.environ.get("MIN_W", 20))
SIMPLIFY_M = float(os.environ.get("SIMPLIFY_M", 0)); NONCANE_MAX_AC = float(os.environ.get("NONCANE_MAX_AC", 2.0)); DROP_PCT = 20.0; DROP_AC = 0.2; TILE_M = float(os.environ.get("TILE_M", 15000)); GAP_TILE_M = float(os.environ.get("GAP_TILE_M", 3000)); N_ANG = int(os.environ.get("N_ANG", 8)); HOLE_AC = 0.05; OPEN_M = 3.0; TAIL_MIN_M2 = 30.0; GAP_M = 3.0; SEP_M = 0.1
ACRE = 4046.8564224; CANE, CASSAVA, NONE_, ND = 1, 281, 0, 65535


def par(fn, *arrays, chunk=2000):
    """a vectorised shapely function on all cores (shapely 2 releases the GIL): arrays split in chunks, run in threads."""
    from concurrent.futures import ThreadPoolExecutor
    from safe_resources import cpus
    n = len(arrays[0])
    if n <= chunk: return fn(*arrays)
    with ThreadPoolExecutor(cpus()) as ex:
        parts = list(ex.map(lambda k: fn(*[a[k:k + chunk] for a in arrays]), range(0, n, chunk)))
    return np.concatenate(parts)


def load_fields(path, bb):
    t = pq.read_table(path, columns=["uid", "master_id", "geometry"])
    g = gpd.GeoDataFrame(t.drop(["geometry"]).to_pandas(), geometry=par(shapely.from_wkb, t["geometry"].to_numpy(), chunk=20000), crs=32648)
    if bb is not None: g = g[g.intersects(box(*bb))]
    g = g.explode(index_parts=False).reset_index(drop=True)
    bad = ~par(shapely.is_valid, g.geometry.values, chunk=20000)
    if bad.any(): g.loc[bad, "geometry"] = shapely.make_valid(g.geometry[bad].values)
    g = g.explode(index_parts=False); g = g[g.geom_type == "Polygon"].reset_index(drop=True)
    if SIMPLIFY_M > 0:                                                           # light simplify (user 26 Oct: faster, shape unchanged)
        s_ = shapely.simplify(g.geometry.values, SIMPLIFY_M, preserve_topology=True); ok = shapely.is_valid(s_) & ~shapely.is_empty(s_) & (abs(shapely.area(s_) / g.area.values - 1) <= 0.02)   # > 2 % area change: keep the original
        g["geometry"] = np.where(ok, s_, g.geometry.values)
    g["fid0"] = np.arange(len(g)); return g, int(bad.sum())


def dedupe(g):
    a = g.area.values; tree = shapely.STRtree(g.geometry.values)
    i, j = tree.query(g.geometry.values); k = i < j; i, j = i[k], j[k]                # bbox candidates (cheap)
    b = shapely.bounds(g.geometry.values); ix = np.clip(np.minimum(b[i, 2], b[j, 2]) - np.maximum(b[i, 0], b[j, 0]), 0, None)
    iy = np.clip(np.minimum(b[i, 3], b[j, 3]) - np.maximum(b[i, 1], b[j, 1]), 0, None)
    ba = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); k = ix * iy >= 0.5 * np.minimum(ba[i], ba[j])     # duplicates / nested copies
    i, j = i[k], j[k]                                                            # share >= half a bbox; edge-touch neighbours do not
    inter = par(lambda a, b: shapely.area(shapely.intersection(a, b)), g.geometry.values[i], g.geometry.values[j])
    iou = inter / (a[i] + a[j] - inter); small = np.where(a[i] <= a[j], i, j)
    drop = set(np.where(a[i] <= a[j], j, i)[iou >= 0.8])                       # duplicate: keep the smaller copy
    nest = inter / np.minimum(a[i], a[j]) >= 0.9
    drop |= set(small[nest & (iou < 0.8)])                                       # nested copy inside a larger field
    return g.drop(index=list(drop)).reset_index(drop=True), len(drop)


def pixel_table(g, cls, tr):
    """per field: pixel centres (x, y) and class of the pixels whose centre lies inside (smaller field wins on overlap)."""
    order = np.argsort(-g.area.values)
    lab = rasterize(((geom, int(k) + 1) for k, geom in zip(order, g.geometry.values[order])), out_shape=cls.shape,
                    transform=tr, fill=0, dtype="int32")
    keep = np.zeros(len(g) + 1, bool); keep[np.unique(lab[cls == CANE])] = True; keep[0] = False   # only fields that hold any cane px
    r, c = np.nonzero(keep[lab])
    f = (lab[r, c] - 1).astype(np.int32); v = cls[r, c]; del lab
    x = tr.c + (c + 0.5) * tr.a; y = tr.f + (r + 0.5) * tr.e; del r, c          # pixel centres, vectorised (no Python lists)
    s = np.argsort(f, kind="stable")
    return f[s], x[s], y[s], v[s]


def shares(v):
    n_c = (v == CANE).sum(); n_v = n_c + (v == CASSAVA).sum() + (v == NONE_).sum()
    return n_c, n_v


def wide(p): return not p.buffer(-MIN_W / 2).is_empty


def best_cut(poly, x, y, v, depth, top=6):
    """straight cuts; returns the kept sugarcane pieces [(polygon, cane_px, valid_px)]. Cut positions are scored on the pixel
    centres first (cumulative counts, cheap); geometry is built only for the `top` best candidates."""
    n_c, n_v = shares(v)
    if n_v and n_c / n_v >= KEEP: return [(poly, n_c, n_v)]
    if depth == 0 or n_c < 4: return []
    mrr = poly.minimum_rotated_rectangle; xs, ys = np.asarray(mrr.exterior.coords).T
    ang = np.arctan2(ys[1] - ys[0], xs[1] - xs[0]); cand = []
    is_c = v == CANE; is_v = is_c | (v == CASSAVA) | (v == NONE_); min_px = MIN_AC * ACRE / 100
    for th in ang + np.arange(N_ANG) * np.pi / N_ANG:                            # field axes first, then every 180/N_ANG degrees
        t = x * np.cos(th) + y * np.sin(th); o = np.argsort(t); ts = t[o]
        cc = np.concatenate([[0], np.cumsum(is_c[o])]); cv = np.concatenate([[0], np.cumsum(is_v[o])]); n = len(ts)
        for cut in np.arange(ts[0] + MIN_W / 2, ts[-1] - MIN_W / 2 + 1e-6, 10.0):
            m = np.searchsorted(ts, cut, side="right")
            for lo_side, (nc, nv, npx) in ((True, (cc[m], cv[m], m)), (False, (cc[-1] - cc[m], cv[-1] - cv[m], n - m))):
                if nc < 4 or npx < min_px: continue
                share = nc / max(nv, 1)
                score = nc - 0.5 * (nv - nc) if share >= KEEP else (nc * share if depth > 1 else -1)
                if score > 0: cand.append((score, th, cut, lo_side))
    best, best_score = [], 0
    for _, th, cut, lo_side in sorted(cand, reverse=True)[:top]:
        big = 1e8                                                                # half plane in the frame (along u, across u)
        half = affinity.rotate(box(-big, -big, cut, big) if lo_side else box(cut, -big, big, big), th, origin=(0, 0), use_radians=True)
        piece = poly.intersection(half)
        if piece.is_empty: continue
        if piece.geom_type != "Polygon":
            ps = [q for q in getattr(piece, "geoms", []) if q.geom_type == "Polygon"]
            if not ps: continue
            piece = max(ps, key=lambda q: q.area)
        if piece.area < MIN_AC * ACRE or not wide(piece): continue
        t = x * np.cos(th) + y * np.sin(th); side = t <= cut if lo_side else t > cut
        got = best_cut(piece, x[side], y[side], v[side], depth - 1, top)
        score = sum(a - 0.5 * (b - a) for _, a, b in got)                       # cane px kept, minus a penalty for non-cane px
        if score > best_score: best, best_score = got, score
    return best


def _cut_job(a): return best_cut(a[0], a[1], a[2], a[3], DEPTH)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fields"); ap.add_argument("aoi"); ap.add_argument("cls"); ap.add_argument("out")
    ap.add_argument("--bbox", nargs=4, type=float); a = ap.parse_args(); t0 = time.time(); log = {}
    aoi = gpd.read_file(a.aoi).to_crs(32648).union_all()
    bb = a.bbox; area_box = box(*bb) if bb else None; clip = aoi.intersection(area_box) if bb else aoi
    ck = a.out.replace(".gpkg", "_precleaned.parquet")
    if os.environ.get("RESUME") == "1" and os.path.exists(ck):                    # restart after the (long) labelling + cut stage
        c = gpd.read_parquet(ck); log = json.load(open(ck + ".json")); print(f"resumed {len(c)} cane polygons from {ck}", flush=True)
        return finish(a, c, log, t0)
    g, log["invalid_repaired"] = load_fields(a.fields, clip.bounds); log["fields_read"] = len(g)
    g, log["duplicates_removed"] = dedupe(g); print(f"fields {len(g)} after dedupe ({time.time() - t0:.0f} s)", flush=True)
    # cut at the AOI boundary
    shapely.prepare(clip); inside = par(lambda a: shapely.within(a, clip), g.geometry.values, chunk=5000); geo = g.geometry.values.copy()
    edge = ~inside & par(lambda a: shapely.intersects(a, clip), geo, chunk=5000)          # only fields on the line are cut,
    geo[edge] = np.array(tmap(lambda p: safe(shapely.intersection, p, shapely.clip_by_rect(clip, *p.bounds)), list(geo[edge])), dtype=object)   # each with
    #                                                                                     the piece of the boundary inside its own bbox (fast)
    g["geometry"] = geo; g["boundary_cut"] = edge; g = g[inside | edge]
    g = g.explode(index_parts=False); g = g[(g.geom_type == "Polygon") & (g.area >= 1)].reset_index(drop=True)
    rows = []; n_keep = 0; jobs = []
    rp = shapely.get_coordinates(par(shapely.point_on_surface, g.geometry.values, chunk=20000))
    x0, y0, x1, y1 = clip.bounds; tx = ((rp[:, 0] - x0) // TILE_M).astype(int); ty = ((rp[:, 1] - y0) // TILE_M).astype(int)
    tid = tx * 10000 + ty; src = rasterio.open(a.cls)
    for t in np.unique(tid):                                                      # labelling per TILE_M tile (memory stays small);
        sub = np.where(tid == t)[0]; gs = g.iloc[sub].reset_index(drop=True)       # a field belongs to the tile of its inner point
        w = from_bounds(*gs.total_bounds, transform=src.transform).round_offsets().round_lengths()
        cls = src.read(1, window=w, boundless=True, fill_value=65534); tr = src.window_transform(w)
        cp = shapely.clip_by_rect(clip, *gs.total_bounds)
        cls = np.where(rasterize([(cp, 1)], out_shape=cls.shape, transform=tr, dtype="uint8") == 1, cls, 65534)   # outside AOI: not counted
        f, x, y, v = pixel_table(gs, cls, tr); starts = np.searchsorted(f, np.arange(len(gs) + 1)); del cls
        for k in range(len(gs)):
            sl = slice(starts[k], starts[k + 1]); vv = v[sl]
            if not (vv == CANE).any(): continue
            n_c, n_v = shares(vv); n_cl = int((vv == ND).sum()); poly = gs.geometry.values[k]
            base = dict(src_fid=int(gs.fid0[k]), uid=gs.uid[k], cloud_pct=round(100 * n_cl / max(len(vv), 1), 1))
            share = n_c / max(n_v, 1); cane_ac = n_c * 100 / ACRE
            non_ac = int(((vv == NONE_) | ((vv == CASSAVA) & (os.environ.get("NONCANE_INCL_CASSAVA") == "1"))).sum()) * 100 / ACRE                              # option B (user 27 Oct): cassava px are cane-like,
            if share >= KEEP or non_ac <= NONCANE_MAX_AC:                                 # only class-0 px count as non-cane here; >= KEEP cane or
                if share < KEEP and share < DROP_PCT / 100 and cane_ac < DROP_AC: continue   # is small (<= NONCANE_MAX_AC): never cut
                rv = "boundary_cut" if gs.boundary_cut[k] else ""                          # (dropped only if < DROP_PCT % AND < DROP_AC)
                if share < KEEP: rv = (rv + ";" if rv else "") + "kept_mixed"
                rows.append(dict(base, geometry=poly, cane_pct=round(100 * n_c / max(n_v, 1), 1), review=rv, exact=not gs.boundary_cut[k])); n_keep += 1
            else: jobs.append((base, poly, x[sl].copy(), y[sl].copy(), vv.copy()))
        del f, x, y, v
    print(f"labelled {len(rows)} whole + {len(jobs)} mixed fields ({time.time() - t0:.0f} s)", flush=True)
    import multiprocessing as mp
    from safe_resources import cpus
    with mp.get_context("fork").Pool(cpus()) as pool:                              # mixed fields: cut search on all cores
        res = pool.map(_cut_job, [(p, xx, yy, vv) for _, p, xx, yy, vv in jobs], chunksize=8)
    n_cut = 0
    for (base, *_), got in zip(jobs, res):
        for piece, a_, b_ in got:
            if a_ / max(b_, 1) < DROP_PCT / 100 and a_ * 100 / ACRE < DROP_AC: continue    # same drop rule for cut pieces
            rows.append(dict(base, geometry=piece, cane_pct=round(100 * a_ / b_, 1), review="cut", exact=False)); n_cut += 1
    c = gpd.GeoDataFrame(rows, crs=32648); log.update(kept_whole=n_keep, cut_pieces=n_cut); print(f"cane polygons {len(c)} ({time.time() - t0:.0f} s)", flush=True)
    c.to_parquet(ck); open(ck + ".json", "w").write(json.dumps(log))            # checkpoint: RESUME=1 restarts from here
    return finish(a, c, log, t0)


def finish(a, c, log, t0):
    c = clean(c, log)
    c["area_ac"] = (c.area / ACRE).round(3); c = c[c.area_ac >= MIN_AC].reset_index(drop=True)
    c.insert(0, "field_id", np.arange(1, len(c) + 1)); c["predicted"] = 1
    tok = c.review.fillna("").str.split(";")                                      # one simple category per polygon
    c["review"] = ["cut" if "cut" in t else "mixed_whole" if "kept_mixed" in t else "whole" for t in tok]
    c = c[["field_id", "predicted", "area_ac", "cane_pct", "cloud_pct", "review", "uid", "src_fid", "geometry"]]
    log.update(qc(c)); log.update(polygons=len(c), area_ac=round(float(c.area_ac.sum()), 3), seconds=round(time.time() - t0))
    if os.path.exists(a.out): os.remove(a.out)
    c.to_file(a.out, layer="sugarcane_fields", driver="GPKG")
    open(a.out.replace(".gpkg", "_log.json"), "w").write(json.dumps(log, indent=1, default=str)); print(json.dumps(log, indent=1, default=str))


def safe(op, a, b):
    """set operation with a GEOS-error retry on a 1 cm, then 10 cm grid (lesson from the earlier field projects)."""
    try: return shapely.make_valid(op(a, b))
    except shapely.errors.GEOSException:
        for gs in (0.01, 0.1):
            try: return shapely.make_valid(op(shapely.make_valid(a), shapely.make_valid(b), grid_size=gs))
            except shapely.errors.GEOSException: pass
        return a


def flag(c, idx, name):
    """only marks the polygon as changed; the review column keeps ONE simple category (whole / mixed_whole / cut, user 27 Oct)."""
    for k in idx:
        c.at[k, "exact"] = False
    return
    for k in idx:
        r = c.at[k, "review"]; c.at[k, "review"] = name if not r else (r if name in r.split(";") else r + ";" + name)
        c.at[k, "exact"] = False


def tmap(fn, items):
    """fn over items in threads on all cores (shapely releases the GIL inside its operations)."""
    from concurrent.futures import ThreadPoolExecutor
    from safe_resources import cpus
    with ThreadPoolExecutor(cpus()) as ex: return list(ex.map(fn, items, chunksize=64))


def clean(c, log):
    geo = c.geometry.values.copy(); tree = shapely.STRtree(geo); i, j = tree.query(geo, predicate="overlaps"); k = i < j; i, j = i[k], j[k]
    ar = shapely.area(geo); big = np.where(ar[i] >= ar[j], i, j); small = np.where(ar[i] >= ar[j], j, i); cut = {}
    for b_, s_ in zip(big, small): cut.setdefault(b_, []).append(s_)            # overlaps: the smaller polygon wins
    keys = list(cut); res = tmap(lambda b_: safe(shapely.difference, geo[b_], shapely.union_all(geo[cut[b_]])), keys)
    for b_, r in zip(keys, res): geo[b_] = r
    c["geometry"] = geo; flag(c, keys, "overlap_fixed"); log["overlaps_fixed"] = len(keys)
    holes = 0
    for k, p in enumerate(c.geometry.values):                                   # holes < HOLE_AC filled
        if p.geom_type == "Polygon" and p.interiors:
            keep = [r for r in p.interiors if shapely.Polygon(r).area >= HOLE_AC * ACRE]
            if len(keep) < len(p.interiors): c.at[k, "geometry"] = shapely.Polygon(p.exterior, keep); holes += 1; flag(c, [k], "holes_filled")
    log["holes_filled"] = holes; desp = 0
    geo = c.geometry.values                                                      # tails / spikes, opening on all cores
    op = par(lambda a: shapely.intersection(shapely.buffer(shapely.buffer(a, -OPEN_M, join_style="mitre"), OPEN_M, join_style="mitre"), a), geo, chunk=500)
    for k, (p, o) in enumerate(zip(geo, op)):
        if o is None or o.is_empty: continue
        if o.geom_type != "Polygon": o = max([q for q in getattr(o, "geoms", [o]) if q.geom_type == "Polygon"] or [p], key=lambda q: q.area)
        if p.area - o.area >= max(TAIL_MIN_M2, 0.02 * p.area) and o.area > 0.5 * p.area: c.at[k, "geometry"] = o; desp += 1; flag(c, [k], "tail_removed")
    log["tails_removed"] = desp
    geo = c.geometry.values.copy(); tree = shapely.STRtree(geo)               # gaps <= 2 x GAP_M between sugarcane neighbours,
    bx = shapely.bounds(geo); x0, y0 = bx[:, 0].min(), bx[:, 1].min(); T = GAP_TILE_M   # per GAP_TILE_M tile on all cores

    def tile_gaps(ij):
        core = shapely.box(x0 + ij[0] * T, y0 + ij[1] * T, x0 + (ij[0] + 1) * T, y0 + (ij[1] + 1) * T)
        idx = tree.query(core.buffer(4 * GAP_M))
        if len(idx) < 2: return []
        U = shapely.union_all(geo[idx]); gp = safe(shapely.difference, U.buffer(GAP_M, join_style="mitre").buffer(-GAP_M, join_style="mitre"), U)
        return [q for q in getattr(gp, "geoms", [gp]) if q.geom_type == "Polygon" and q.area >= 1 and core.contains(q.representative_point())]
    nx, ny = int((bx[:, 2].max() - x0) // T) + 1, int((bx[:, 3].max() - y0) // T) + 1
    parts = [q for ps in tmap(tile_gaps, [(i, j) for i in range(nx) for j in range(ny)]) for q in ps]

    def owner(q):
        qb = q.buffer(0.5); cand = tree.query(qb, predicate="intersects")
        if len(cand) < 2: return -1                                              # a bay of one field, not a gap between fields
        return int(cand[int(np.argmax(shapely.length(shapely.intersection(shapely.boundary(geo[cand]), qb))))])
    own = tmap(owner, parts); grp = {}
    for q, m in zip(parts, own):
        if m >= 0: grp.setdefault(m, []).append(q)
    keys = list(grp); res = tmap(lambda m: safe(shapely.union, geo[m], shapely.union_all(grp[m])), keys)
    for m, r in zip(keys, res): geo[m] = r
    c["geometry"] = geo; flag(c, keys, "gap_filled"); log["gaps_filled"] = int(sum(len(v) for v in grp.values()))
    geo = c.geometry.values.copy(); orig = c.geometry.values; tree = shapely.STRtree(geo); i, j = tree.query(geo, predicate="dwithin", distance=2 * SEP_M)
    k = i != j; near = {}
    for a_, b_ in zip(i[k], j[k]): near.setdefault(a_, []).append(b_)
    keys = list(near)                                                            # SEP_M on each side between neighbours, all cores
    res = tmap(lambda a_: safe(shapely.difference, geo[a_], shapely.union_all(orig[near[a_]]).buffer(SEP_M, join_style="mitre")), keys)
    for a_, r in zip(keys, res): geo[a_] = r
    c["geometry"] = geo; flag(c, keys, "separated"); log["separated"] = len(near)
    c = c.explode(index_parts=False); c = c[c.geom_type == "Polygon"].reset_index(drop=True)
    ch = ~c["exact"].astype(bool).values; g = c.geometry.values.copy()      # only CHANGED polygons: 1 cm grid + drop micro holes
    g[ch] = shapely.set_precision(g[ch], 0.01)                                 # (untouched fields keep the exact original geometry)
    g[ch] = [shapely.Polygon(p.exterior, [r for r in p.interiors if shapely.Polygon(r).area >= HOLE_AC * ACRE]) if p.geom_type == "Polygon" else p for p in g[ch]]
    g[ch] = shapely.make_valid(g[ch]); c["geometry"] = g; c = c.explode(index_parts=False); c = c[c.geom_type == "Polygon"]
    return c.reset_index(drop=True)


def qc(c):
    g = c.geometry.values; tree = shapely.STRtree(g)
    i, j = tree.query(g, predicate="intersects"); k = i < j
    holes = sum(sum(1 for r in p.interiors if shapely.Polygon(r).area < HOLE_AC * ACRE) for p in g)
    return dict(qc_invalid=int((~shapely.is_valid(g)).sum()), qc_multipart=int((shapely.get_num_geometries(g) > 1).sum()),
                qc_intersect_or_touch_pairs=int(k.sum()), qc_small_holes=holes,
                qc_duplicate_geoms=int(len(g) - len(set(shapely.to_wkb(g)))))


if __name__ == "__main__":
    main()

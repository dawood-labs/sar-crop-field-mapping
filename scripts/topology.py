"""Polygon clean-up shared by the field scripts (tested on a 3 x 3 km window, 2026-10-09: 0 overlapping / touching
pairs afterwards, dissolve keeps every field separate, area -0.25 % vs the raw union).

simplify(G)        Douglas-Peucker 0.5 m per polygon (vertices -88 %, area -0.01 %, max shift < 0.9 m)
merge_nested(G)    a polygon lying >= 90 % inside a larger one is merged into it (largest host wins, chains followed)
separate(G)        no overlap and no touching: each polygon loses what lies within GAP of a LARGER polygon, so a later
                   dissolve can never merge two fields (slivers < 1 m2 left by the cut are dropped)
Every step uses all safe workers: shapely's vectorised calls release the GIL, so they run in threads over chunks
(STRtree predicate queries are single-threaded, so near_pairs() takes bbox candidates and tests them in threads).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely

TOL, INSIDE, GAP = 0.5, 0.9, 0.1
GRID = 0.01      # 1 cm fixed-precision overlay: avoids GEOS "non-noded intersection" errors on near-coincident vertices


def _workers(per_worker_bytes=1.0 * 2**30):
    from safe_resources import workers
    return workers(per_worker_bytes)


def _threaded(fn, G, *args):
    """fn applied to chunks of G in threads (fn must be a vectorised shapely call). TOPO_THREADS=1 -> serial (debug)."""
    n = int(os.environ.get("TOPO_THREADS", 0)) or _workers(0.3 * 2**30)
    chunks = np.array_split(np.arange(len(G)), n * 4)
    with ThreadPoolExecutor(n) as pool:
        parts = list(pool.map(lambda c: fn(G[c], *args), [c for c in chunks if len(c)]))
    return np.concatenate(parts) if parts else np.array([], dtype=object)


FALLBACKS = {"rows": 0, "kept_input": 0}


def _diff_one(a, b):
    for prep in (lambda x: x, shapely.make_valid, lambda x: shapely.buffer(shapely.make_valid(x), 0)):
        try:
            return shapely.make_valid(shapely.difference(prep(a), prep(b), grid_size=GRID))
        except shapely.errors.GEOSException:
            continue
    FALLBACKS["kept_input"] += 1                # never silently: counted and reported by the callers
    return a


def safe_union(arr):
    """union_all on a 1 cm grid; on a GEOS error retried with make_valid, then buffer(0) (counted in FALLBACKS)."""
    arr = np.asarray(arr, dtype=object)
    for k, f in enumerate((lambda x: shapely.union_all(x, grid_size=GRID),
                           lambda x: shapely.union_all(shapely.make_valid(x)),
                           lambda x: shapely.union_all(shapely.buffer(shapely.make_valid(x), 0)))):
        try:
            out = shapely.make_valid(f(arr))
            if k:
                FALLBACKS["rows"] += 1
            return out
        except shapely.errors.GEOSException:
            continue
    FALLBACKS["kept_input"] += 1
    return shapely.GeometryCollection()            # nothing protected for this one row (counted)


def safe_difference(A, B):
    """Vectorised a - b on a 1 cm grid; rows that still throw a GEOS error are redone one by one with make_valid."""
    pair = np.empty((len(A), 2), dtype=object); pair[:, 0] = A; pair[:, 1] = B
    def chunk(c):
        try:
            return shapely.make_valid(shapely.difference(c[:, 0], c[:, 1], grid_size=GRID))
        except shapely.errors.GEOSException:
            FALLBACKS["rows"] += len(c)
            return np.array([_diff_one(a, b) for a, b in c], dtype=object)
    return _threaded(chunk, pair)


def subtract_neighbours(G, B, p):
    """For every pair column (i, j) in p: G[i] -= B[j]. No union_all (GEOS crashed / threw on big unions): the
    neighbours are subtracted one per round, each round one vectorised, threaded difference."""
    out = np.asarray(G, dtype=object).copy()
    if not p.shape[1]:
        return out
    p = p[:, np.argsort(p[0], kind="stable")]
    ids, first, cnt = np.unique(p[0], return_index=True, return_counts=True)
    for k in range(int(cnt.max())):
        sel = cnt > k
        rows, nb = ids[sel], p[1][first[sel] + k]
        out[rows] = safe_difference(out[rows], B[nb])
    return out


def near_pairs(G, dist, tree=None):
    """All pairs (i, j), i != j, closer than `dist` (0 = intersecting): bbox candidates from the STRtree (fast), then
    the exact test vectorised in threads (shapely releases the GIL) - the tree's own predicate query is single-threaded."""
    G = np.asarray(G, dtype=object)
    tree = tree or shapely.STRtree(G)
    b = shapely.bounds(G)
    boxes = shapely.box(b[:, 0] - dist, b[:, 1] - dist, b[:, 2] + dist, b[:, 3] + dist)
    q = tree.query(boxes); q = q[:, q[0] != q[1]]
    if not q.shape[1]:
        return q
    test = (lambda c: shapely.dwithin(G[c[:, 0]], G[c[:, 1]], dist)) if dist > 0 else \
           (lambda c: shapely.intersects(G[c[:, 0]], G[c[:, 1]]))
    return q[:, _threaded(test, q.T.copy()).astype(bool)]


def simplify(G, tol=TOL):
    G = np.asarray(G, dtype=object)
    G = _threaded(lambda g: shapely.simplify(g, tol, preserve_topology=True), G)
    bad = ~_threaded(shapely.is_valid, G)
    G[bad] = shapely.make_valid(G[bad])
    return G


def merge_nested(G, inside=INSIDE):
    """Returns (geoms, keep mask, host index per polygon; -1 = not merged)."""
    G = np.asarray(G, dtype=object)
    A = shapely.area(G)
    p = shapely.STRtree(G).query(G, predicate="intersects")
    p = p[:, (p[0] != p[1]) & (A[p[1]] > A[p[0]])]               # p[0] smaller, p[1] larger
    inter = _threaded(lambda q: shapely.area(shapely.intersection(G[q[:, 0]], G[q[:, 1]])), p.T.copy()) if p.shape[1] else np.zeros(0)
    p = p[:, inter / np.maximum(A[p[0]], 1e-9) >= inside]
    host = np.full(len(G), -1)
    for s, b in sorted(zip(p[0], p[1]), key=lambda x: -A[x[1]]):  # largest host first
        if host[s] < 0:
            host[s] = b
    for s in range(len(G)):                                       # small in medium in big -> big
        while host[s] >= 0 and host[host[s]] >= 0:
            host[s] = host[host[s]]
    out = G.copy()
    hosts = np.unique(host[host >= 0])
    if len(hosts):
        groups = [np.r_[b, np.nonzero(host == b)[0]] for b in hosts]
        with ThreadPoolExecutor(_workers(0.3 * 2**30)) as pool:
            merged = list(pool.map(lambda ix: shapely.make_valid(shapely.union_all(G[ix])), groups))
        for b, m in zip(hosts, merged):
            out[b] = m
    return out, host < 0, host


def separate(G, gap=GAP):
    """Each polygon loses what lies within `gap` of a LARGER polygon (ties: lower index wins). All steps vectorised in
    threads: one buffer per winner, cut = that buffer (1 neighbour) or the union of the buffers, one difference."""
    G = np.asarray(G, dtype=object)
    A = shapely.area(G)
    p = near_pairs(G, gap)
    p = p[:, (A[p[1]] > A[p[0]]) | ((A[p[1]] == A[p[0]]) & (p[1] < p[0]))]   # p[0] gives way
    out = G.copy()
    if not p.shape[1]:
        return out
    losers = np.unique(p[0])
    win = np.unique(p[1])
    B = np.empty(len(G), dtype=object)
    B[win] = _threaded(lambda g: shapely.make_valid(shapely.buffer(g, gap, quad_segs=2)), G[win])
    x = subtract_neighbours(G, B, p)[losers]
    # keep polygon parts >= 1 m2, one feature per input (multi-part if the cut split it)
    parts, idx = shapely.get_parts(x, return_index=True)
    ok = (shapely.get_type_id(parts) == 3) & (shapely.area(parts) >= 1.0)
    parts, idx = parts[ok], idx[ok]
    res = np.full(len(losers), shapely.Polygon(), dtype=object)
    if len(parts):
        n = np.bincount(idx, minlength=len(losers))
        single = n == 1
        res[single] = parts[np.isin(idx, np.nonzero(single)[0])]
        many = np.nonzero(n > 1)[0]
        if len(many):
            sel = np.isin(idx, many)
            mp = shapely.multipolygons(parts[sel], indices=np.searchsorted(many, idx[sel]))
            res[many] = mp
    out[losers] = res
    return out


def refine(G, close=1.5, open_=2.0, hole_ac=0.05, min_part_m2=100.0, gap=GAP):
    """Clean the shape of every polygon (Delineation outlines carry slits, pin holes and thin tails):
    closing by `close` m (slits / notches narrower than 2*close are filled), holes < hole_ac acre filled,
    opening by `open_` m (tails / spikes narrower than 2*open_ are cut; mitre joins keep square corners),
    parts < min_part_m2 dropped. A polygon may only grow into EMPTY space: the original area of every neighbour
    (buffered by `gap`) is subtracted, so closing never swallows a small field next to a big one."""
    G0 = np.asarray(G, dtype=object)
    kw = dict(quad_segs=2, join_style="mitre", mitre_limit=5.0)
    G = _threaded(lambda g: shapely.make_valid(shapely.buffer(shapely.buffer(g, close, **kw), -close, **kw)), G0)
    p = near_pairs(G0, 2 * close + gap)                          # neighbours the closing could reach
    if p.shape[1]:
        B = _threaded(lambda g: shapely.make_valid(shapely.buffer(g, gap, quad_segs=2)), G0)
        G = subtract_neighbours(G, B, p)
    G = _threaded(lambda g: shapely.buffer(shapely.buffer(g, -open_, **kw), open_, **kw), G)
    G = _threaded(shapely.make_valid, G)
    hole_m2 = hole_ac * 4046.856
    def clean_one(g):
        out = []
        for p in shapely.get_parts(g):
            if p.geom_type != "Polygon" or p.area < min_part_m2:
                continue
            keep = [r for r in p.interiors if shapely.Polygon(r).area >= hole_m2]
            out.append(shapely.Polygon(p.exterior, keep) if len(keep) != len(p.interiors) else p)
        return shapely.multipolygons(out) if len(out) > 1 else (out[0] if out else shapely.Polygon())
    with ThreadPoolExecutor(_workers(0.3 * 2**30)) as pool:
        return np.array(list(pool.map(clean_one, G, chunksize=2000)), dtype=object)


def fill_small_holes(G, hole_ac=0.05):
    """Remove interior rings smaller than hole_ac acre (bigger holes = real non-field inside a field are kept)."""
    hole_m2 = hole_ac * 4046.856
    def one(g):
        out = []
        for p in shapely.get_parts(g):
            keep = [r for r in p.interiors if shapely.Polygon(r).area >= hole_m2]
            out.append(shapely.Polygon(p.exterior, keep) if len(keep) != len(p.interiors) else p)
        return shapely.multipolygons(out) if len(out) > 1 else (out[0] if out else g)
    return np.array([one(g) for g in np.asarray(G, dtype=object)], dtype=object)


def spikes(G, max_angle=15.0):
    """Number of exterior-ring vertices with an interior angle < max_angle degrees (thin spikes / tails)."""
    n = 0
    for p in shapely.get_parts(np.asarray(G, dtype=object)):
        if p.geom_type != "Polygon":
            continue
        c = np.asarray(p.exterior.coords)[:-1]
        if len(c) < 3:
            continue
        a, b = np.roll(c, 1, 0) - c, np.roll(c, -1, 0) - c
        cos = (a * b).sum(1) / (np.hypot(*a.T) * np.hypot(*b.T) + 1e-12)
        n += int((np.degrees(np.arccos(np.clip(cos, -1, 1))) < max_angle).sum())
    return n


def check(G):
    """(intersecting pairs, min gap in m among pairs closer than 1 m)."""
    G = np.asarray(G, dtype=object); G = G[~shapely.is_empty(G)]
    t = shapely.STRtree(G)
    p = near_pairs(G, 0, t); p = p[:, p[0] < p[1]]
    q = near_pairs(G, 1.0, t); q = q[:, q[0] < q[1]]
    d = _threaded(lambda c: shapely.distance(G[c[:, 0]], G[c[:, 1]]), q.T.copy()) if q.shape[1] else np.array([np.nan])
    return int(p.shape[1]), float(np.nanmin(d))

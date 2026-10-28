"""Province set-up: the units to process + their S1 tracks -> plan on S3 (<province prefix>/_plan/plan.json), AOIs on S3.

units = "districts"  GAUL level-2 districts of the province (data/source/districts.parquet) + gap AOIs: province polygon minus
                     the districts shrunk by 20 m (the overlap is resolved by the mosaic), pieces >= 1 acre, grouped by the
                     tracks of the nearest district.
units = "grid"       square tiles of grid_km on a fixed 10 m UTM 48N grid (EPSG:32648; tile edges on multiples of the tile
                     size, so tiles never overlap and every pixel belongs to one tile), clipped to the province polygon;
                     tiles with < 1 acre inside are dropped. No gap AOIs are needed.
S1 tracks per unit from Earth Engine: the model's RO164 DES / RO099 ASC when they cover >= 98 % of the unit, else the
best-covering track of that pass (aliased to the model's names in the composites).
`--local` (JupyterHub pod only): the plan is built from the existing site folders of the province instead.
A plan that already exists on S3 is kept unless the stage "prepare" is forced (changing the units of a half-done province
would orphan its finished sites).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import s3state as S3
import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import site_paths as SP  # noqa: E402
from .config import PROJ, Params, Province, Site, plan_uri, save_plan
from .site import ensure_shared, log, site_prefix, site_root

ACRE = 4046.8564224
GAP_TRACKS = {"province_le": (62, 99), "province5": (164, 99), "province1": (164, 99), "province7": (62, 172),
              "site154_2026": (164, 99), "site144_2026": (164, 26)}   # tracks the existing gap runs used


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    head = s.split("_")[0]
    return {"muang": "mueang"}.get(head, head) + s[len(head):]


# ------------------------------------------------------------------ S1 tracks
_EE_READY = False


def s1_coverage(geom_4326, p: Params) -> dict:
    """{(orbit, 'DES'|'ASC'): fraction of the unit covered} over the season (same filter as download.py)."""
    import sys
    import ee
    sys.path.insert(0, str(PROJ / "scripts"))
    from download import init_ee_from_env                      # noqa: E402
    global _EE_READY
    if not _EE_READY:
        init_ee_from_env(); _EE_READY = True
    g = ee.Geometry(geom_4326.__geo_interface__)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(g).filterDate(p.season_start, p.season_end)
           .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    ro = col.aggregate_array("relativeOrbitNumber_start").getInfo(); ps = col.aggregate_array("orbitProperties_pass").getInfo()
    area = g.area(1); out = {}
    for r, ps_ in sorted({(r, x) for r, x in zip(ro, ps)}):
        c = col.filter(ee.Filter.eq("relativeOrbitNumber_start", r)).filter(ee.Filter.eq("orbitProperties_pass", ps_))
        out[(r, ps_[:3])] = c.geometry().intersection(g, 1).area(1).divide(area).getInfo()
    return out


def pick(cov: dict, pas: str, model: int) -> int:
    if cov.get((model, pas), 0) >= 0.98:
        return model
    best = max(((c, r) for (r, x), c in cov.items() if x == pas), default=(0, None))
    if best[1] is None:
        raise RuntimeError(f"no {pas} Sentinel-1 track over the unit")
    return best[1]


def tracks_for(geom_32648, p: Params) -> tuple[int, int]:
    import geopandas as gpd
    g4326 = gpd.GeoSeries([geom_32648], crs=32648).to_crs(4326).iloc[0]
    cov = s1_coverage(g4326.simplify(0.0005), p)
    return pick(cov, "DES", 164), pick(cov, "ASC", 99)


# ------------------------------------------------------------------ units
def _province_geoms(p: Params):
    import geopandas as gpd
    ensure_shared(p)
    pv = gpd.read_parquet(PROJ / "data/source/provinces.parquet")
    pv = pv[pv.ADM1_EN.str.replace(" ", "_") == p.province].to_crs(32648)
    if pv.empty:
        raise RuntimeError(f"{p.province} is not in provinces.parquet (use '_' for spaces, e.g. province5)")
    dist = gpd.read_parquet(PROJ / "data/source/districts.parquet")
    dist = dist[dist.province.str.replace(" ", "_") == p.province].to_crs(32648)
    return pv.geometry.union_all(), dist


def _write_aoi(p: Params, name: str, geoms):
    import geopandas as gpd
    d = site_root(p, name) / "data/aoi"; d.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"name": [name] * len(geoms)}, geometry=list(geoms), crs=32648).to_crs(4326).to_file(d / "aoi.shp")


def units_districts(p: Params) -> list[Site]:
    import geopandas as gpd
    prov_geom, dist = _province_geoms(p)
    if dist.empty:
        raise RuntimeError(f"{p.province}: no districts in districts.parquet - use units=grid")
    sites, rows = [], []
    for _, r in dist.iterrows():
        n = f"{slug(r.ADM2_NAME)}_{p.year}"
        des, asc = tracks_for(r.geometry, p)
        _write_aoi(p, n, [r.geometry]); sites.append(Site(n, p.province, des, asc, "district")); rows.append((des, asc))
        log(f"{n}: {r.geometry.area / ACRE:,.0f} ac, DES RO{des:03d} ASC RO{asc:03d}")
    gap = prov_geom.difference(dist.geometry.buffer(-20).union_all())
    pieces = gpd.GeoDataFrame(geometry=[gap], crs=32648).explode(index_parts=False)
    pieces = pieces[pieces.area >= ACRE].reset_index(drop=True)
    if len(pieces):
        ref = gpd.GeoDataFrame({"des": [a for a, _ in rows], "asc": [b for _, b in rows]}, geometry=dist.geometry.values, crs=32648)
        near = gpd.sjoin_nearest(pieces, ref, how="left"); near = near[~near.index.duplicated()]
        groups = list(near.groupby(["des", "asc"]))
        for (d, a), grp in groups:
            n = f"{p.province.lower()}_gap_{p.year}" if len(groups) == 1 else f"{p.province.lower()}_gap{a}_{p.year}"
            _write_aoi(p, n, grp.geometry.values); sites.append(Site(n, p.province, int(d), int(a), "gap"))
            log(f"{n}: {len(grp)} pieces, {grp.area.sum() / ACRE:,.0f} ac, DES RO{int(d):03d} ASC RO{int(a):03d}")
    return sites


def units_grid(p: Params) -> list[Site]:
    """Square tiles on a fixed grid (edges on multiples of grid_km in EPSG:32648), clipped to the province; slivers below
    min_tile_acres are merged into the touching tile with the most area (one download per tile is the fixed cost)."""
    import math
    from shapely.geometry import box
    prov_geom, _ = _province_geoms(p)
    t = p.grid_km * 1000
    xmin, ymin, xmax, ymax = prov_geom.bounds
    cells = []                                                          # [(i, j), geometry], north to south, west to east
    for j in range(math.floor(ymax / t), math.floor(ymin / t) - 1, -1):
        for i in range(math.floor(xmin / t), math.floor(xmax / t) + 1):
            c = box(i * t, j * t, (i + 1) * t, (j + 1) * t).intersection(prov_geom)
            if not c.is_empty and c.area >= ACRE:
                cells.append([(i, j), c])
    for small in sorted([c for c in cells if c[1].area < p.min_tile_acres * ACRE], key=lambda c: c[1].area):
        if small not in cells:
            continue
        nb = [c for c in cells if c is not small and c[1].buffer(1).intersects(small[1])]
        if nb:
            big = max(nb, key=lambda c: c[1].area)
            big[1] = big[1].union(small[1]); cells.remove(small)
    sites = []
    for k, ((i, j), cell) in enumerate(cells, 1):
        n = f"{p.province.lower()}_g{k:03d}_{p.year}"
        des, asc = tracks_for(cell, p)
        _write_aoi(p, n, [cell]); sites.append(Site(n, p.province, des, asc, "grid"))
        log(f"{n}: tile ({i},{j}) {cell.area / ACRE:,.0f} ac, DES RO{des:03d} ASC RO{asc:03d}")
    return sites


def _manifest_tracks(root: Path):
    m = root / "data/manifest/download_manifest.csv"
    if not m.exists():
        return None
    t = pd.read_csv(m, usecols=["sensor", "track"]).query("sensor == 's1'").track.unique()
    des = [int(x[2:5]) for x in t if x.endswith("DES")]; asc = [int(x[2:5]) for x in t if x.endswith("ASC")]
    return (des[0], asc[0]) if len(des) == 1 and len(asc) == 1 else None


def units_local(p: Params) -> list[Site]:
    """JupyterHub pod: the site folders that exist (gap tiles excluded); tracks from the plan file / manifest."""
    folder = SP.SITES / p.year / p.province
    plan = PROJ / "sites" / f"{p.province.lower()}_plan.txt"
    planned = {}
    if plan.exists():
        for line in plan.read_text().split("\n"):
            if line.strip():
                n, d, a = line.split()[:3]; planned[f"{n}_{p.year}"] = (int(d), int(a))
    sites = []
    for d in sorted(folder.glob(f"*_{p.year}")):
        if not (d / "data/aoi/aoi.shp").exists() or re.search(r"_gap\d*_t\d\d_", d.name):
            continue
        kind = "gap" if "_gap" in d.name else "district"
        tr = planned.get(d.name) or _manifest_tracks(d) or GAP_TRACKS.get(d.name) or \
            (GAP_TRACKS.get(p.province) if kind == "gap" else None) or (164, 99)
        sites.append(Site(d.name, p.province, tr[0], tr[1], kind))
    return sites


def prepare(p: Params, local: bool = False) -> Province:
    if S3.read_json(plan_uri(p)) is not None and not p.forced("prepare"):
        from .config import load_plan
        prov = load_plan(p)
        log(f"{p.province}: plan already on S3 ({len(prov.sites)} sites, units={prov.units}) - kept (force=prepare to rebuild)")
        return prov
    units = "districts" if local else p.units
    sites = units_local(p) if local else units_grid(p) if units == "grid" else units_districts(p)
    prov = Province(p.province, units, sites)
    if not p.dry_run:
        for s in sites:                                # AOIs to S3: a fresh pod fetches them from there
            S3.upload_dir(site_root(p, s.name), site_prefix(p, s.name), ["data/aoi"])
        save_plan(prov, p)
    log(f"{p.province}: {len(sites)} units ({units}) -> {plan_uri(p)}")
    return prov

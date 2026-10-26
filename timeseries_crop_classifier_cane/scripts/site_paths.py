"""Where district folders live (user, 22 Oct 2026): sites/<year>/<Province>/<site>/, e.g. sites/2026/province7/site45_2026.
Folder names keep the year suffix (site45_2026) because every CSV / review file / model json uses them as site keys.
Shared folders (layer2, user_labels, testset_v1, benchmark, reviews, index CSVs) stay directly in sites/.

  site_dir("site45_2026")          -> Path of the district folder (old-style "sites/site45_2026" also works)
  all_sites("*_2026")                 -> sorted list of district folders matching the pattern (any province)
  python scripts/site_paths.py place  -> moves any district folder still lying directly in sites/ into sites/<year>/<Province>/
  python scripts/site_paths.py list   -> prints year, province, site
  python scripts/site_paths.py dir <site>  -> prints the folder (for shell drivers: S=$(python scripts/site_paths.py dir site45_2026))
"""
from __future__ import annotations

import re, sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SITES = PROJ / "sites"


def all_sites(pattern: str = "*_20??") -> list[Path]:
    return sorted((p for p in SITES.glob(f"20??/*/{pattern}") if p.is_dir()), key=lambda p: p.name)


def site_dir(name) -> Path:
    """District folder from a site name or any old/new path to it. Unknown names fall back to sites/<name> (new site)."""
    p = Path(str(name))
    if p.is_absolute() and p.exists(): return p
    if (PROJ / p).exists() and p.parts[:1] == ("sites",) and len(p.parts) > 2: return (PROJ / p).resolve()
    base = p.name
    hits = [h for h in SITES.glob(f"20??/*/{base}") if h.is_dir()]
    if hits: return hits[0]
    return SITES / base


def flat_rel(path) -> str:
    """Local path -> project-relative key with the OLD flat layout (sites/<site>/...). GCS / S3 keys keep that layout."""
    r = Path(path).resolve().relative_to(PROJ)
    if r.parts[:1] == ("sites",) and len(r.parts) >= 4 and re.fullmatch(r"20\d\d", r.parts[1]): r = Path("sites", *r.parts[3:])
    return r.as_posix()


def from_flat(rel) -> Path:
    """Old flat key (sites/<site>/...) -> local path in the new layout; anything else -> PROJ / rel."""
    parts = Path(str(rel)).parts
    if parts[:1] == ("sites",) and len(parts) > 1 and re.search(r"_20\d\d", parts[1]): return site_dir(parts[1]).joinpath(*parts[2:])
    return PROJ / rel


def province_mask(prov: str, shape, transform, crs):
    """True inside the official province polygon (data/source/provinces.parquet, ADM1_EN = folder name with spaces) on a grid.
    District AOIs (GAUL) do not match the province edge, so province products are clipped with this (user, 23 Oct 2026)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    pv = gpd.read_parquet(PROJ / "data/source/provinces.parquet")
    hit = pv[pv.ADM1_EN.str.replace(" ", "_").str.lower() == prov.lower()]
    if hit.empty: raise ValueError(f"province {prov} not in provinces.parquet")
    g = hit.to_crs(crs).geometry
    return rasterize([(x, 1) for x in g], out_shape=shape, transform=transform, dtype="uint8").astype(bool)


def year_of(site: str) -> str:
    m = re.search(r"_(20\d\d)(?:_|$)", site)
    return m.group(1) if m else "other"


def province_of(folder: Path) -> str:
    import geopandas as gpd
    shp = sorted(folder.glob("data/aoi/*.shp"))
    if not shp: raise FileNotFoundError(f"{folder}: no data/aoi/*.shp, cannot find the province")
    pv = gpd.read_parquet(PROJ / "data/source/provinces.parquet")
    pt = gpd.read_file(shp[0]).to_crs(pv.crs).union_all().representative_point()
    hit = pv[pv.contains(pt)]
    if hit.empty: hit = pv.iloc[[pv.distance(pt).argmin()]]
    return hit.ADM1_EN.iloc[0].replace(" ", "_")


def place():
    for f in sorted(SITES.glob("*_20??")):
        if not f.is_dir(): continue
        if not list(f.glob("data/aoi/*.shp")): print(f.name, "skipped: no data/aoi/*.shp yet"); continue
        dst = SITES / year_of(f.name) / province_of(f) / f.name
        dst.parent.mkdir(parents=True, exist_ok=True); f.rename(dst); print(f.name, "->", dst.relative_to(PROJ))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "place": place()
    elif cmd == "dir": print(site_dir(sys.argv[2]))
    else:
        for s in all_sites(): print(s.parts[-3], s.parts[-2], s.name)

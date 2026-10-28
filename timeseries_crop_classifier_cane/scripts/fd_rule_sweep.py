"""Sweep of the 'cut as little as possible' rule for the field-delineation sugarcane polygons (user, 27 Oct 2026).
Per delineated field (pixel centres of the static class raster, cloud px not counted):
  share >= KEEP                                -> keep whole
  else: non-cane area <= X and cane >= CANE_MIN -> keep whole (no cut)
        non-cane area > X                       -> straight cuts (fd_cane_fields.best_cut), keep the cane pieces
        else (cane < CANE_MIN)                  -> drop
For every X: fields kept whole / cut / dropped, non-cane acres taken in (commission), cane acres left out (omission), total acres.
usage: python scripts/fd_rule_sweep.py <fields.parquet> <aoi.gpkg> <class.tif> --bbox xmin ymin xmax ymax  -> reports/fd_rule_sweep_<tag>.csv
"""
import argparse, multiprocessing as mp, os, time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import box

import fd_cane_fields as F
from safe_resources import cpus

XS = [0.0, 0.25, 0.5, 0.81, 1.0, 1.66, 2.5]; CANE_MIN = float(os.environ.get("CANE_MIN", 0.2)); PX = 100 / F.ACRE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fields"); ap.add_argument("aoi"); ap.add_argument("cls")
    ap.add_argument("--bbox", nargs=4, type=float, required=True); ap.add_argument("--tag", default="test10km"); a = ap.parse_args(); t0 = time.time()
    clip = gpd.read_file(a.aoi).to_crs(32648).union_all().intersection(box(*a.bbox))
    g, _ = F.load_fields(a.fields, clip.bounds); g, _ = F.dedupe(g)
    inside = shapely.within(g.geometry.values, clip); geo = g.geometry.values.copy(); edge = ~inside & shapely.intersects(geo, clip)
    geo[edge] = shapely.intersection(geo[edge], clip); g["geometry"] = geo; g = g[inside | edge]
    g = g.explode(index_parts=False); g = g[(g.geom_type == "Polygon") & (g.area >= 1)].reset_index(drop=True)
    with rasterio.open(a.cls) as r:
        w = from_bounds(*g.total_bounds, transform=r.transform).round_offsets().round_lengths(); cls = r.read(1, window=w, boundless=True, fill_value=65534); tr = r.window_transform(w)
    cls = np.where(rasterize([(clip, 1)], out_shape=cls.shape, transform=tr, dtype="uint8") == 1, cls, 65534)
    f, x, y, v = F.pixel_table(g, cls, tr); st = np.searchsorted(f, np.arange(len(g) + 1))
    rec, jobs = [], []
    for k in range(len(g)):
        vv = v[st[k]:st[k + 1]]
        if not (vv == F.CANE).any(): continue
        n_c, n_v = F.shares(vv); rec.append(dict(k=k, nc=n_c, nv=n_v, ac=g.geometry.values[k].area / F.ACRE))
        if not (n_v and n_c / n_v >= F.KEEP): jobs.append((g.geometry.values[k], x[st[k]:st[k + 1]], y[st[k]:st[k + 1]], vv))
    with mp.get_context("fork").Pool(cpus()) as pool: cuts = pool.map(F._cut_job, jobs, chunksize=8)
    d = pd.DataFrame(rec); mixed = d[~(d.nc / d.nv.clip(lower=1) >= F.KEEP)].reset_index(drop=True)
    mixed["cut_nc"] = [sum(p[1] for p in c) for c in cuts]; mixed["cut_nv"] = [sum(p[2] for p in c) for c in cuts]; mixed["cut_n"] = [len(c) for c in cuts]
    whole = d[d.nc / d.nv.clip(lower=1) >= F.KEEP]; base_c, base_n = whole.nc.sum(), (whole.nv - whole.nc).sum()
    cane_total = d.nc.sum(); out = []
    for X in XS:
        non = (mixed.nv - mixed.nc) * PX; cane = mixed.nc * PX
        keep = (non <= X) & (cane >= CANE_MIN); cut = ~keep & (non > X); drop = ~keep & ~cut
        kc = base_c + mixed.nc[keep].sum() + mixed.cut_nc[cut].sum(); kn = base_n + (mixed.nv - mixed.nc)[keep].sum() + (mixed.cut_nv - mixed.cut_nc)[cut].sum()
        out.append(dict(X_noncane_ac=X, fields_whole=len(whole) + int(keep.sum()), fields_cut=int(((mixed.cut_n > 0) & cut).sum()), fields_dropped=int(drop.sum() + ((mixed.cut_n == 0) & cut).sum()),
                        cane_in_ac=round(kc * PX), noncane_in_ac=round(kn * PX), cane_left_out_ac=round((cane_total - kc) * PX),
                        noncane_share_pct=round(100 * kn / max(kc + kn, 1), 1), raster_cane_ac=round(cane_total * PX)))
    r = pd.DataFrame(out); os.makedirs("../reports", exist_ok=True); r.to_csv(f"../reports/fd_rule_sweep_{a.tag}.csv", index=False)
    mixed.to_parquet(f"../reports/fd_rule_sweep_{a.tag}_mixed.parquet"); print(r.to_string(index=False)); print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

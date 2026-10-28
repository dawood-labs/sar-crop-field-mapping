"""New sugarcane fields where the static map has sugarcane but the delineation has NO field (user, 26 Oct 2026):
the outline comes from the Google satellite basemap (as the delineation itself did), not from the 10 m raster.

1. orphan sugarcane = static-class px == 1 not covered by any delineated field (after the same dedupe as fd_cane_fields.py),
   8-connected blobs >= MIN_BLOB_AC
2. per blob: Google satellite tiles (zoom ZOOM, ~0.57 m) around the blob, SAM (vit_h, segment_anything) with positive point
   prompts inside the blob; the mask with the best overlap with the blob is taken as the field outline
3. outline -> EPSG:32648, within NEAR_M of the blob, field-shaped (>= MIN_W wide, Polsby-Popper >= MIN_PP: roads and strips
   between fields are dropped), minus every delineated field (never overlaps an existing field), sugarcane share from the raster
   pixel centres: >= KEEP kept, else the same straight-cut rule as fd_cane_fields.py; pieces < MIN_AC dropped
Output: <out.gpkg> layers new_fields (review = new_basemap) and blobs (the orphan raster blobs, for checking).
usage: python scripts/fd_basemap_fields.py <fields.parquet> <class.tif> <out.gpkg> --bbox xmin ymin xmax ymax
       env SAM_CKPT (default ~/Downloads/sam_vit_h_4b8939.pth), ZOOM=18, MIN_BLOB_AC=0.2, TILE_DIR (cache)
"""
import argparse, io, math, os, time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import rasterio
import requests
import shapely
from PIL import Image
from pyproj import Transformer
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import shape

import fd_cane_fields as F

ZOOM = int(os.environ.get("ZOOM", 18)); MIN_BLOB_AC = float(os.environ.get("MIN_BLOB_AC", 0.2)); MARGIN = 60.0; NEAR_M = 30.0; MIN_PP = 0.18
CKPT = os.path.expanduser(os.environ.get("SAM_CKPT", "~/Downloads/sam_vit_h_4b8939.pth"))
TILE_DIR = os.environ.get("TILE_DIR", "data/basemap_tiles")
URL = "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"; R = 6378137.0; SIZE = 2 * math.pi * R / 2 ** ZOOM
TO3857 = Transformer.from_crs(32648, 3857, always_xy=True); TO32648 = Transformer.from_crs(3857, 32648, always_xy=True)


def tile(x, y):
    p = os.path.join(TILE_DIR, str(ZOOM), str(x), f"{y}.jpg")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        r = requests.get(URL.format(s=(x + y) % 4, x=x, y=y, z=ZOOM), timeout=30, headers={"User-Agent": "Mozilla/5.0"}); r.raise_for_status()
        open(p + ".tmp", "wb").write(r.content); os.replace(p + ".tmp", p)
    return np.asarray(Image.open(p).convert("RGB"))


def basemap(bounds_3857):
    x0, y0, x1, y1 = bounds_3857
    cs = range(int((x0 + math.pi * R) // SIZE), int((x1 + math.pi * R) // SIZE) + 1)
    rs = range(int((math.pi * R - y1) // SIZE), int((math.pi * R - y0) // SIZE) + 1)
    with ThreadPoolExecutor(32) as ex: ims = list(ex.map(lambda rc: tile(rc[1], rc[0]), [(r, c) for r in rs for c in cs]))
    img = np.vstack([np.hstack(ims[i * len(cs):(i + 1) * len(cs)]) for i in range(len(rs))])
    tr = Affine(SIZE / 256, 0, cs[0] * SIZE - math.pi * R, 0, -SIZE / 256, math.pi * R - rs[0] * SIZE)
    return img, tr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fields"); ap.add_argument("cls"); ap.add_argument("out")
    ap.add_argument("--bbox", nargs=4, type=float, required=True); a = ap.parse_args(); t0 = time.time()
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    from safe_resources import cpus
    torch.set_num_threads(cpus())
    g, _ = F.load_fields(a.fields, a.bbox); g, _ = F.dedupe(g)
    with rasterio.open(a.cls) as r:
        w = from_bounds(*a.bbox, transform=r.transform).round_offsets().round_lengths(); cls = r.read(1, window=w); tr = r.window_transform(w)
    cov = rasterize(((x, 1) for x in g.geometry.values), out_shape=cls.shape, transform=tr, dtype="uint8") == 1
    lab, n = ndimage.label((cls == F.CANE) & ~cov, structure=np.ones((3, 3)))
    sz = np.bincount(lab.ravel()); keep = [k for k in range(1, n + 1) if sz[k] * 100 / F.ACRE >= MIN_BLOB_AC]
    blobs = [shapely.union_all([shape(s) for s, v in shapes((lab == k).astype("uint8"), mask=lab == k, transform=tr)]) for k in keep]
    print(f"{len(blobs)} orphan blobs >= {MIN_BLOB_AC} ac ({time.time() - t0:.0f} s)", flush=True)
    sam = SamPredictor(sam_model_registry["vit_h"](checkpoint=CKPT)); tree = shapely.STRtree(g.geometry.values); out = []
    for k, (bk, blob) in enumerate(zip(keep, blobs)):
        b3 = shapely.ops.transform(TO3857.transform, blob); img, itr = basemap(b3.buffer(MARGIN).bounds)
        rr, cc = np.nonzero(ndimage.binary_erosion(lab == bk) if ndimage.binary_erosion(lab == bk).sum() >= 3 else lab == bk)
        xs, ys = rasterio.transform.xy(tr, rr, cc); px, py = TO3857.transform(np.asarray(xs), np.asarray(ys))
        col, row = ~itr * (np.asarray(px), np.asarray(py)); pts = np.c_[col, row]
        pts = pts[np.linspace(0, len(pts) - 1, min(5, len(pts))).astype(int)]
        sam.set_image(img); masks, scores, _ = sam.predict(point_coords=pts, point_labels=np.ones(len(pts)), multimask_output=True)
        bmask = rasterize([(b3, 1)], out_shape=img.shape[:2], transform=itr, dtype="uint8") == 1
        iou = [(m & bmask).sum() / max((m | bmask).sum(), 1) for m in masks]; m = masks[int(np.argmax(iou))]
        poly = shapely.union_all([shape(s) for s, v in shapes(m.astype("uint8"), mask=m, transform=itr)])
        poly = shapely.make_valid(shapely.ops.transform(TO32648.transform, poly))
        near = tree.query(poly, predicate="intersects")
        if len(near): poly = F.safe(shapely.difference, poly, shapely.union_all(g.geometry.values[near]))
        poly = F.safe(shapely.intersection, poly, blob.buffer(NEAR_M))              # the basemap edge only near the raster blob
        parts = [p for p in getattr(poly, "geoms", [poly]) if p.geom_type == "Polygon" and p.area >= F.MIN_AC * F.ACRE and p.intersects(blob)
                 and F.wide(p) and 4 * math.pi * p.area / p.length ** 2 >= MIN_PP]   # field-shaped: >= 20 m wide, not a road / strip
        for p in parts:
            lb = rasterize([(p, 1)], out_shape=cls.shape, transform=tr, dtype="uint8") == 1; r2, c2 = np.nonzero(lb)
            x2, y2 = rasterio.transform.xy(tr, r2, c2); v2 = cls[r2, c2]
            for piece, nc, nv in F.best_cut(p, np.asarray(x2), np.asarray(y2), v2, F.DEPTH):
                out.append(dict(geometry=piece, cane_pct=round(100 * nc / nv, 1), blob_ac=round(blob.area / F.ACRE, 3), sam_iou=round(float(max(iou)), 2), review="new_basemap"))
        print(f"blob {k + 1}/{len(blobs)} {blob.area / F.ACRE:.2f} ac -> {len(parts)} part(s), iou {max(iou):.2f} ({time.time() - t0:.0f} s)", flush=True)
    n_ = gpd.GeoDataFrame(out, crs=32648) if out else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=32648)
    if len(n_): n_["area_ac"] = (n_.area / F.ACRE).round(3)
    if os.path.exists(a.out): os.remove(a.out)
    n_.to_file(a.out, layer="new_fields", driver="GPKG")
    gpd.GeoDataFrame(dict(blob_ac=[round(b.area / F.ACRE, 3) for b in blobs]), geometry=blobs, crs=32648).to_file(a.out, layer="blobs", driver="GPKG")
    print(f"new fields {len(n_)}, {n_.area.sum() / F.ACRE if len(n_) else 0:.1f} ac, raster orphan {sum(b.area for b in blobs) / F.ACRE:.1f} ac ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

"""Split a big, sparse site (province gap AOIs: a province-sized grid with few AOI pixels) into tile sub-sites (26 Oct 2026).
The per-district scripts read whole bands (apply_site reads all features at once), so a 14k x 10k px gap grid exceeded the pod's
40 GB memory limit and the container was restarted. Each tile = TILE x TILE px of the parent grid, cropped to its AOI pixels, as a
normal site folder next to the parent: <parent>_tNN_2026/data/{grid (index.tif + grid.json), aoi, stacks, manifest}.
Pixel ids, grid and values are copied unchanged (windows of the parent files), so a tile is the same as a small district.
usage: python scripts/split_site.py <site>        env TILE=3000   prints the tile site names
"""
import json, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window, transform as wtransform
from shapely.geometry import box

import site_paths as SP
import safe_resources

TILE = int(os.environ.get("TILE", 3000))


def main(name):
    S = SP.site_dir(name); base = S.name.removesuffix("_2026")
    with rasterio.open(S / "data/grid/index.tif") as g:
        H, W = g.height, g.width; tiles = []
        for r0 in range(0, H, TILE):
            for c0 in range(0, W, TILE):
                w = Window(c0, r0, min(TILE, W - c0), min(TILE, H - r0)); a = g.read(2, window=w) > 0
                if not a.any(): continue
                rr, cc = np.where(a.any(1))[0], np.where(a.any(0))[0]
                tiles.append(Window(c0 + cc[0], r0 + rr[0], cc[-1] - cc[0] + 1, rr[-1] - rr[0] + 1))
        gprof, gtr = g.profile, g.transform
    grid = json.loads((S / "data/grid/grid.json").read_text()); aoi = gpd.read_file(S / "data/aoi/aoi.shp")
    stacks = sorted((S / "data/stacks").glob("*.tif")); out = []
    for k, w in enumerate(tiles, 1):
        T = S.parent / f"{base}_t{k:02d}_2026"; tr = wtransform(w, gtr); out.append(T.name)
        for d in ("grid", "aoi", "stacks", "manifest", "logs"): (T / "data" / d if d != "logs" else T / d).mkdir(parents=True, exist_ok=True)
        if (T / "data/stacks/stacks.csv").exists(): continue                     # already split
        with rasterio.open(S / "data/grid/index.tif") as g, rasterio.open(T / "data/grid/index.tif", "w", **dict(gprof, width=w.width, height=w.height, transform=tr)) as d:
            d.write(g.read(window=w)); d.descriptions = g.descriptions
        (T / "data/grid/grid.json").write_text(json.dumps(dict(grid, x0=tr.c, y0=tr.f, width=int(w.width), height=int(w.height), transform=list(tr)[:6]), indent=2))
        bb = box(*rasterio.windows.bounds(w, gtr)); a = aoi.clip(bb)
        a.to_file(T / "data/aoi/aoi.shp")
        for f in (S / "data/manifest").iterdir(): shutil.copy2(f, T / "data/manifest" / f.name)

        def crop(f):
            with rasterio.open(f) as s:
                p = dict(s.profile, width=w.width, height=w.height, transform=tr)
                with rasterio.open(T / "data/stacks" / f.name, "w", **p) as d:
                    d.write(s.read(window=w)); d.descriptions = s.descriptions
                    for i in range(1, s.count + 1): d.update_tags(i, **s.tags(i))
                    d.update_tags(**s.tags())
        with ThreadPoolExecutor(min(len(stacks), safe_resources.io_threads())) as ex: list(ex.map(crop, stacks))
        shutil.copy2(S / "data/stacks/stacks.csv", T / "data/stacks/stacks.csv")   # last: marks the tile complete
        print(f"{T.name}: {w.width} x {w.height} px, AOI {int((rasterio.open(T / 'data/grid/index.tif').read(2) > 0).sum()) / 40.468564224:,.0f} ac", flush=True)
    (S / "tiles.txt").write_text("\n".join(out) + "\n"); print(" ".join(out))


if __name__ == "__main__":
    main(sys.argv[1])

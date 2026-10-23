"""Elevation and slope on the project grid: data/ancillary/dem.tif (band 1 elevation m, band 2 slope degrees).

Source: Copernicus GLO-30 DSM (COPERNICUS/DEM/GLO30, ~30 m, a surface model: tree/building tops, not bare ground),
resampled bilinear to the 10 m grid. Slope is computed locally from the resampled elevation (np.gradient, 10 m).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import download as dl  # noqa: E402  (grid, fetch, write_tif, Earth Engine auth)


def main():
    import ee, json
    dl.init_ee(dl.CFG, high_volume=True)
    grid = json.loads((PROJ / "data/grid/grid.json").read_text())
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    dem = col.mosaic().setDefaultProjection(col.first().projection()).resample("bilinear")
    z = dl.fetch(dem, ["DEM"], np.float32, grid, -9999.0)[0]
    z[z == -9999.0] = np.nan
    gy, gx = np.gradient(z, dl.RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    out = PROJ / "data/ancillary/dem.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    dl.write_tif(out, np.stack([z, slope]), grid, ["elevation_m", "slope_deg"], np.nan,
                 {"source": "COPERNICUS/DEM/GLO30 (DSM), bilinear to 10 m; slope from np.gradient"})
    with np.errstate(all="ignore"):
        print(f"elevation {np.nanpercentile(z, [1, 50, 99]).round(1)} m, slope p50/p95 {np.nanpercentile(slope, [50, 95]).round(1)} deg")


if __name__ == "__main__":
    main()

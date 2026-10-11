"""Extra Sentinel-1 features for rice v6: data/features/s1_features_v6_extra.tif (6 bands, used next to v4's 84).

Why (user check of v5, 2026-10-09): sugarcane mapped as rice (prob 90) because its ratoon starts growing in JANUARY
(VH -21 dB in Jan -> -15.5 dB in Mar) or because it is cut in Feb-Apr (green until Jan, then a sharp VH drop).
v4's vhdry_trend only compares Apr-May with Feb-Mar, so both patterns slip through.
Bands per track (RO164_DES, RO099_ASC; VH, 3x3 window, dB; dates with >= 10 mm IMERG rain in the previous 24 h dropped):
  <track>_vh_janmar_rise   mean VH March 2026 minus mean VH January 2026            (ratoon regrowth from January)
  <track>_vh_cut           mean VH Dec 2025-Jan 2026 minus lowest VH Feb-Apr 2026   (late cut = cane harvest)
  <track>_vh_dry_amp       highest VH Feb-15 May 2026 minus lowest VH Dec 2025-Mar 2026 (any dry-season growth)
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
from s1_features_v4 import load, window  # noqa: E402

OUT = PROJ / "data/features/s1_features_v6_extra.tif"


def track_bands(track, rain):
    dates, dry, _, w3 = load(track, "VH", rain)
    mean = lambda a, b: window(dates, dry, w3, a, b, lambda q: np.nanmean(q, 0))  # noqa: E731
    vmin = lambda a, b: window(dates, dry, w3, a, b, lambda q: np.nanmin(q, 0))   # noqa: E731
    vmax = lambda a, b: window(dates, dry, w3, a, b, lambda q: np.nanmax(q, 0))   # noqa: E731
    with np.errstate(all="ignore"):
        out = {f"{track}_vh_janmar_rise": mean("2026-03-01", "2026-03-31") - mean("2026-01-01", "2026-01-31"),
               f"{track}_vh_cut": mean("2025-12-01", "2026-01-31") - vmin("2026-02-01", "2026-04-30"),
               f"{track}_vh_dry_amp": vmax("2026-02-01", "2026-05-15") - vmin("2025-12-01", "2026-03-31")}
    del w3
    return out


def main():
    t0 = time.time()
    rain = pd.read_csv(PROJ / "data/rain/rain_s1_acquisitions.csv"); rain["date"] = rain.date.astype(str)
    bands = {}
    for t in ("RO164_DES", "RO099_ASC"):          # one track at a time: ~6-7 GB peak, well under the 16 GB request
        bands.update(track_bands(t, rain))
        print(f"{t} done {time.time() - t0:.0f} s", flush=True)
    names = sorted(bands)
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        prof = r.profile.copy()
    prof.update(count=len(names), dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(OUT, "w", **prof) as dst:
        dst.write(np.stack([bands[k].astype("float32") for k in names]))
        for i, k in enumerate(names, 1):
            dst.set_band_description(i, k)
    print(f"{len(names)} bands -> {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

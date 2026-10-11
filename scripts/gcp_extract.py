"""Step 1 of the GCP EDA: extract every time series and summary features for the 100 ground-sample points.

Input : data/gcps/ground_samples/ground_samples.shp (Name, Crop), point geometry, EPSG:4326
Output: data/gcps/eda/gcp_series.pkl   per point: S2 table (1x1), S1 table (3x3 window, linear mean), rain
        data/gcps/eda/gcp_features.csv one row per point (features below)

Features (NDVI from cloud-free dates only, 1x1 pixel; S1 from a 3x3 window to cut speckle):
  ndvi_<period>      median clear NDVI in each period (SepOct25, NovDec25, JanFeb26, MarApr26, MayJun26, JulAug26, SepOct26)
  n_clear_<period>   clear observations used
  ndvi_ws_min/max, date_ws_min/max   Whittaker curve (lambda 1, d 2) extremes and their dates
  vh_<track>_<period>, vv_<track>_<period>  median dB per period
  vh_<track>_min_JunAug, date: lowest VH in the flooding window (rice transplanting / standing water)
  flood_frac_<track>  share of May-Aug 2026 S1 dates with VH < -20 dB
  het_ndvi            median within-5x5 NDVI std on clear dry-season dates (high = point near a field edge / mixed)
  dup_pixel           another point falls in the same 10 m pixel
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import pickle, sys, time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import explorer as ex  # noqa: E402

OUT = PROJ / "data/gcps/eda"
PERIODS = {"SepOct25": ("2025-09-01", "2025-10-31"), "NovDec25": ("2025-11-01", "2025-12-31"),
           "JanFeb26": ("2026-01-01", "2026-02-28"), "MarApr26": ("2026-03-01", "2026-04-30"),
           "MayJun26": ("2026-05-01", "2026-06-30"), "JulAug26": ("2026-07-01", "2026-08-31"),
           "SepOct26": ("2026-09-01", "2026-10-08")}


def in_period(dates, p):
    a, b = PERIODS[p]
    return (dates >= a) & (dates <= b)


def features(g, px1, px3):
    f = {}
    s2 = px1["s2"]
    clr = s2[s2.clear]
    for p in PERIODS:
        m = in_period(clr.date, p)
        f[f"ndvi_{p}"] = clr.ndvi[m].median() if m.any() else np.nan
        f[f"n_clear_{p}"] = int(m.sum())
    f["n_clear"] = len(clr)
    dd, z = ex.whittaker(s2.date, s2.ndvi.values, s2.clear.astype(float).values, 1.0, 2)
    ws = pd.Series(z, index=dd)
    ws = ws[(ws.index >= "2025-09-15") & (ws.index <= "2026-08-27")]          # avoid edge extrapolation
    f["ndvi_ws_min"], f["date_ws_min"] = ws.min(), ws.idxmin().strftime("%Y-%m-%d")
    f["ndvi_ws_max"], f["date_ws_max"] = ws.max(), ws.idxmax().strftime("%Y-%m-%d")
    dry = clr[(clr.date >= "2026-01-01") & (clr.date <= "2026-04-30")]
    f["ndvi_dry_min"] = dry.ndvi.min() if len(dry) else np.nan
    f["ndvi_dry_max"] = dry.ndvi.max() if len(dry) else np.nan
    s1 = px3["s1"]
    for t, g1 in s1.groupby("track"):
        k = t.split("_")[0]
        for p in PERIODS:
            m = in_period(g1.date, p)
            f[f"vh_{k}_{p}"] = g1.VH[m].median() if m.any() else np.nan
            f[f"vv_{k}_{p}"] = g1.VV[m].median() if m.any() else np.nan
        ja = g1[(g1.date >= "2026-06-01") & (g1.date <= "2026-08-31")]
        f[f"vh_{k}_min_JunAug"] = ja.VH.min()
        f[f"date_vh_{k}_min_JunAug"] = ja.date[ja.VH.idxmin()].strftime("%Y-%m-%d") if len(ja) else None
        ma = g1[(g1.date >= "2026-05-01") & (g1.date <= "2026-08-31")]
        f[f"flood_frac_{k}"] = float((ma.VH < -20).mean()) if len(ma) else np.nan
        f[f"vh_{k}_mean"], f[f"vh_{k}_std"] = g1.VH.mean(), g1.VH.std()
        f[f"vv_{k}_mean"], f[f"vhvv_{k}_mean"] = g1.VV.mean(), g1.VH_minus_VV.mean()
    return f


def heterogeneity(row, col, s2_clear_dates_idx):
    """Median (over clear dry-season dates) of the NDVI std inside the 5x5 window around the point."""
    a = ex._pixel("S2_NDVI", row, col, 2) / 1e4                    # (dates, 25)
    if not len(s2_clear_dates_idx):
        return np.nan
    with np.errstate(all="ignore"):
        return float(np.nanmedian(np.nanstd(a[s2_clear_dates_idx], axis=1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file(PROJ / "data/gcps/ground_samples/ground_samples.shp")
    g["gcp_id"] = [f"PD{int(n):03d}" for n in g.Name]
    t0 = time.time()
    rows, series = [], {}
    for r in g.itertuples():
        lon, lat = r.geometry.x, r.geometry.y
        px1 = ex.load_pixel(lon=lon, lat=lat, radius=0)
        px3 = ex.load_pixel(lon=lon, lat=lat, radius=1)
        s2 = px1["s2"]
        dry_idx = np.where(s2.clear & (s2.date >= "2026-01-01") & (s2.date <= "2026-04-30"))[0]
        f = {"gcp_id": r.gcp_id, "name": r.Name, "crop": r.Crop, "lon": lon, "lat": lat,
             "row": px1["row"], "col": px1["col"], "pixel_id": px1["pixel_id"], **features(r, px1, px3),
             "het_ndvi": heterogeneity(px1["row"], px1["col"], dry_idx)}
        dem = ex._ds(PROJ / "data/ancillary/dem.tif")
        from rasterio.windows import Window
        z = dem.read(window=Window(px1["col"], px1["row"], 1, 1)).ravel()
        f["elev_m"], f["slope_deg"] = float(z[0]), float(z[1])
        rows.append(f)
        series[r.gcp_id] = {"s2": s2, "s1_3x3": px3["s1"], "s1_1x1": px1["s1"], "rain": px1["rain_daily"],
                            "row": px1["row"], "col": px1["col"], "crop": r.Crop, "lon": lon, "lat": lat}
    df = pd.DataFrame(rows)
    df["dup_pixel"] = df.duplicated("pixel_id", keep=False)
    from scipy.spatial import cKDTree
    xy = np.c_[df.col * 10, df.row * 10]
    d, _ = cKDTree(xy).query(xy, k=2)
    df["nearest_point_m"] = d[:, 1]
    df.to_csv(OUT / "gcp_features.csv", index=False)
    pickle.dump(series, open(OUT / "gcp_series.pkl", "wb"))
    print(f"{len(df)} points in {time.time() - t0:.0f} s; duplicates in a pixel: {int(df.dup_pixel.sum())}; "
          f"nearest neighbour median {df.nearest_point_m.median():.0f} m, min {df.nearest_point_m.min():.0f} m")


if __name__ == "__main__":
    main()

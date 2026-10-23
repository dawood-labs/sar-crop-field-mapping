"""Rainfall over the AOI, to flag Sentinel-1 dates taken on wet soil / wet canopy.

Products (Earth Engine, AOI mean, all in mm):
  CHIRPS v2 daily   UCSB-CHG/CHIRPS/DAILY            ~5.5 km, station-blended, best daily totals; ends 2026-08-31
  IMERG V07         NASA/GPM_L3/IMERG_V07            ~11 km, every 30 min (mm/h -> x0.5 per step)
  GSMaP v8 (gauge)  JAXA/GPM_L3/GSMaP/v8/operational ~11 km, hourly, band hourlyPrecipRateGC
  ERA5-Land         ECMWF/ERA5_LAND/HOURLY           ~9 km, hourly model, total_precipitation_hourly (m -> mm)
TRMM ended in 2019 and is not usable for 2025-26.

Outputs (data/rain/):
  rain_daily.csv          one row per UTC day: chirps, imerg, gsmap, era5 (NaN where a product has no data yet)
  rain_s1_acquisitions.csv one row per Sentinel-1 file in the download manifest: rain in the 6/12/24/48 h before the
                          acquisition time (IMERG, GSMaP, ERA5) + CHIRPS on the acquisition day and the day before
  chirps_daily.tif        CHIRPS daily rasters at native 0.05 deg over the AOI box (one band per day, band name = date)
"""
from __future__ import annotations

import sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sar_pipeline.auth import init_ee  # noqa: E402

START, END = "2025-09-01", "2026-10-09"
OUT = PROJ / "data/rain"
CFG = {"_root": str(__import__("pathlib").Path(__file__).resolve().parents[1]),
       "auth": {"key_file": __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json"), "project": __import__("os").environ.get("EE_PROJECT", "")}}
# id, band, factor to mm per image, native scale (m)
HOURLY = {"imerg": ("NASA/GPM_L3/IMERG_V07", "precipitation", 0.5, 11000),
          "gsmap": ("JAXA/GPM_L3/GSMaP/v8/operational", "hourlyPrecipRateGC", 1.0, 11000),
          "era5": ("ECMWF/ERA5_LAND/HOURLY", "total_precipitation_hourly", 1000.0, 11000)}
WINDOWS_H = [6, 12, 24, 48]


def aoi_geom():
    import ee, geopandas as gpd
    g = gpd.read_file(PROJ / "data/aoi/PhanomDongRak_AOI.shp").to_crs(4326)
    return ee.Geometry(g.geometry.unary_union.__geo_interface__)


def window_sum(cid, band, factor, scale, aoi, t0, t1):
    """Server-side: AOI-mean rain (mm) between two ee.Dates; null if the product has no image in the window."""
    import ee
    col = ee.ImageCollection(cid).filterDate(t0, t1).select(band)
    total = col.sum().multiply(factor).reduceRegion(ee.Reducer.mean(), aoi, scale, bestEffort=True).get(band)
    return ee.Algorithms.If(col.size().gt(0), total, None)


def daily_table(aoi):
    import ee
    days = pd.date_range(START, pd.Timestamp(END) - pd.Timedelta(days=1), freq="D")
    rows = {d.strftime("%Y-%m-%d"): {"date": d.strftime("%Y-%m-%d")} for d in days}
    chunks = [days[i:i + 60] for i in range(0, len(days), 60)]          # keep each request small
    for name, (cid, band, f, sc) in {"chirps": ("UCSB-CHG/CHIRPS/DAILY", "precipitation", 1.0, 5566), **HOURLY}.items():
        for ch in chunks:
            fc = ee.FeatureCollection([ee.Feature(None, {
                "date": d.strftime("%Y-%m-%d"),
                "mm": window_sum(cid, band, f, sc, aoi, ee.Date(d.strftime("%Y-%m-%d")), ee.Date(d.strftime("%Y-%m-%d")).advance(1, "day"))})
                for d in ch])
            for ft in fc.getInfo()["features"]:
                p = ft["properties"]; rows[p["date"]][name] = p.get("mm")
        print(f"  daily {name} done", flush=True)
    return pd.DataFrame(rows.values())


def s1_table(aoi):
    import ee
    man = pd.read_csv(PROJ / "data/manifest/download_manifest.csv")
    s1 = man[man.sensor == "s1"].copy()
    # acquisition start time from the first scene id: S1C_IW_GRDH_1SDV_20250906T225959_...
    s1["time_utc"] = s1.scenes.str.split(";").str[0].str.split("_").str[4].map(
        lambda t: datetime.strptime(t, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc))
    feats = []
    for r in s1.itertuples():
        t = ee.Date(r.time_utc.isoformat())
        props = {"track": r.track, "date": r.date, "time_utc": r.time_utc.strftime("%Y-%m-%d %H:%M"),
                 "time_local": (r.time_utc + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M")}
        for name, (cid, band, f, sc) in HOURLY.items():
            for h in WINDOWS_H:
                props[f"{name}_{h}h"] = window_sum(cid, band, f, sc, aoi, t.advance(-h, "hour"), t)
        d = ee.Date(r.time_utc.strftime("%Y-%m-%d"))
        props["chirps_day"] = window_sum("UCSB-CHG/CHIRPS/DAILY", "precipitation", 1.0, 5566, aoi, d, d.advance(1, "day"))
        props["chirps_prev_day"] = window_sum("UCSB-CHG/CHIRPS/DAILY", "precipitation", 1.0, 5566, aoi, d.advance(-1, "day"), d)
        feats.append(ee.Feature(None, props))
    out = []
    for i in range(0, len(feats), 30):
        out += [f["properties"] for f in ee.FeatureCollection(feats[i:i + 30]).getInfo()["features"]]
    return pd.DataFrame(out).sort_values(["track", "date"])


def chirps_raster(aoi):
    import ee, rasterio
    col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(START, END)
    names = col.aggregate_array("system:index").getInfo()                       # YYYYMMDD
    img = col.toBands().rename(names)
    b = aoi.bounds(1).buffer(5600).bounds(1).coordinates().getInfo()[0]
    x0, y1 = min(p[0] for p in b), max(p[1] for p in b)
    res = 0.05
    x0, y1 = np.floor(x0 / res) * res, np.ceil(y1 / res) * res
    w = int(np.ceil((max(p[0] for p in b) - x0) / res)); h = int(np.ceil((y1 - min(p[1] for p in b)) / res))
    arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                 "grid": {"dimensions": {"width": w, "height": h}, "crsCode": "EPSG:4326",
                                          "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0,
                                                              "shearY": 0, "scaleY": -res, "translateY": y1}}})
    stack = np.stack([arr[n] for n in names]).astype("float32")
    with rasterio.open(OUT / "chirps_daily.tif", "w", driver="GTiff", width=w, height=h, count=len(names),
                       dtype="float32", crs="EPSG:4326", transform=rasterio.transform.from_origin(x0, y1, res, res),
                       compress="deflate") as dst:
        dst.write(stack)
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)
        dst.update_tags(units="mm/day", source="UCSB-CHG/CHIRPS/DAILY")
    print(f"  chirps raster {w}x{h} cells, {len(names)} days", flush=True)


def main():
    import ee
    init_ee(CFG, high_volume=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); aoi = aoi_geom()
    chirps_raster(aoi)
    daily_table(aoi).round(2).to_csv(OUT / "rain_daily.csv", index=False)
    s1_table(aoi).round(2).to_csv(OUT / "rain_s1_acquisitions.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

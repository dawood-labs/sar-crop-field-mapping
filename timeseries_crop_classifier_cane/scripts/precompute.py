"""Precompute everything the pixel explorer notebook needs, so a pixel plots in about a second.

Outputs
-------
data/stacks/S2_NDVI.tif      NDVI x 10000 (int16, -32768 = no data), all S2 dates, raw (clouds included)
data/stacks/S2_CLEAR.tif     1 = clear, 0 = cloudy/shadow, 255 = no data. Clear means ALL of:
                               Cloud Score+ cs_cdf >= 60, SCL not in {0 no data, 1 saturated, 3 shadow, 8, 9 cloud,
                               10 cirrus}, MSK_CLASSI_OPAQUE == 0, MSK_CLASSI_CIRRUS == 0, B4 > 0
data/stacks/S2_NDVI_WS.tif   Whittaker-smoothed NDVI x 10000 at the S2 dates (lambda = 1.0, d = 2), fitted on a DAILY
                               grid with weight 1 on clear days and 0 elsewhere (so uneven gaps are handled correctly)
data/stacks/s2_stretch.json  2nd/98th percentile of B5, B3, B2 over clear AOI pixels (fixed chip stretch)
data/rain/imerg_daily.tif            IMERG daily rain (mm), native 0.1 deg, AOI box, one band per day
data/rain/s1_rain24h_imerg.tif       IMERG rain in the 24 h before every Sentinel-1 acquisition (band = TRACK_DATE)
data/rain/s1_rain24h_gsmap.tif       same with GSMaP (gauge-corrected)
data/rain/rain_s1_acquisitions.csv   AOI-mean rain before each S1 acquisition (6/12/24/48 h, several products)
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
ST = PROJ / "data/stacks"
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sar_pipeline.resources import detect_resources  # noqa: E402

LAMBDA, ORDER, CS_MIN = 1.0, 2, 60
BAD_SCL = [0, 1, 3, 8, 9, 10]
CLDPRB_MAX = 10                                  # fallback when Cloud Score+ is not yet published for a date


def read(name):
    with rasterio.open(ST / f"{name}.tif") as r:
        return r.read(), list(r.descriptions), r.profile.copy()


def write(name, arr, dates, prof, nodata, tags):
    prof.update(count=arr.shape[0], dtype=arr.dtype, nodata=nodata, predictor=2, interleave="pixel",
                compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    tmp = ST / f"{name}.tmp"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        for k, d in enumerate(dates, 1):
            dst.set_band_description(k, d)
        dst.update_tags(**tags)
    tmp.replace(ST / f"{name}.tif")


# ---------------------------------------------------------------- Whittaker (batched over pixels)
def whittaker_daily(y, w, lam=LAMBDA, d=ORDER):
    """Solve (W + lam D'D) z = W y for many series at once. y, w: (n_days, n_px) float64. d must be 2.
    Symmetric pentadiagonal LDL^T, vectorised across pixels (the time loop is only n_days long)."""
    assert d == 2
    n, P = y.shape
    a0 = np.full(n, 6.0); a0[[0, -1]] = 1; a0[[1, -2]] = 5
    a1 = np.full(n - 1, -4.0); a1[[0, -1]] = -2          # A[i+1, i]
    a2 = np.ones(n - 2)                                   # A[i+2, i]
    A0 = w + lam * a0[:, None]
    dd = np.empty((n, P)); l1 = np.zeros((n, P)); l2 = np.zeros((n, P))
    for i in range(n):
        if i >= 2:
            l2[i] = lam * a2[i - 2] / dd[i - 2]
        if i >= 1:
            l1[i] = (lam * a1[i - 1] - (l2[i] * dd[i - 2] * l1[i - 1] if i >= 2 else 0)) / dd[i - 1]
        dd[i] = A0[i] - (l1[i] ** 2 * dd[i - 1] if i >= 1 else 0) - (l2[i] ** 2 * dd[i - 2] if i >= 2 else 0)
    z = w * y
    for i in range(1, n):                                 # forward: L u = b
        z[i] -= l1[i] * z[i - 1] + (l2[i] * z[i - 2] if i >= 2 else 0)
    z /= dd                                               # D v = u
    for i in range(n - 2, -1, -1):                        # backward: L^T x = v
        z[i] -= l1[i + 1] * z[i + 1] + (l2[i + 2] * z[i + 2] if i + 2 < n else 0)
    return z


def s2_layers():
    t0 = time.time()
    b4, dates, prof = read("S2_B4"); b8, _, _ = read("S2_B8")
    with np.errstate(all="ignore"):
        ndvi = (b8.astype("float32") - b4) / (b8.astype("float32") + b4)
    valid = (b4 > 0) & (b8 > 0)
    nd = np.where(valid, np.round(ndvi * 10000), -32768).astype("int16")
    write("S2_NDVI", nd, dates, prof.copy(), -32768, {"variable": "NDVI x 10000", "formula": "(B8-B4)/(B8+B4)"})
    del b8
    cs, _, _ = read("S2_cs_cdf"); scl, _, _ = read("S2_SCL")
    op, _, _ = read("S2_MSK_CLASSI_OPAQUE"); ci, _, _ = read("S2_MSK_CLASSI_CIRRUS")
    # dates whose Cloud Score+ is not published yet (cs all 65535): fall back to ESA cloud probability <= CLDPRB_MAX
    cs_missing = np.array([not ((cs[k] != 65535) & valid[k]).any() for k in range(len(dates))])
    cs_ok = (cs != 65535) & (cs >= CS_MIN)
    if cs_missing.any():
        pr, _, _ = read("S2_MSK_CLDPRB")
        cs_ok[cs_missing] = pr[cs_missing] <= CLDPRB_MAX
        del pr
        print(f"  no Cloud Score+ yet for {[dates[k] for k in np.nonzero(cs_missing)[0]]}: ESA MSK_CLDPRB <= {CLDPRB_MAX} used", flush=True)
    clear = valid & cs_ok & ~np.isin(scl, BAD_SCL) & (op == 0) & (ci == 0)
    clr = np.where(valid, clear, 255).astype("uint8")
    write("S2_CLEAR", clr, dates, prof.copy(), 255,
          {"rule": f"cs_cdf>={CS_MIN} (or MSK_CLDPRB<={CLDPRB_MAX} on dates without Cloud Score+) & SCL not in {BAD_SCL} "
                   f"& MSK_CLASSI_OPAQUE==0 & MSK_CLASSI_CIRRUS==0 & B4>0"})
    del cs, scl, op, ci, b4, cs_ok
    print(f"  NDVI + CLEAR stacks {time.time() - t0:.0f} s", flush=True)

    if os.environ.get("SKIP_WS", "1") == "1":       # standing-cane project: QC uses clear-date NDVI only
        print("  Whittaker stack skipped (SKIP_WS=1)", flush=True)
    else:
        # Whittaker on a daily grid, in pixel chunks across threads (numpy releases the GIL in the big array ops)
        t0 = time.time()
        day0 = pd.Timestamp(dates[0])
        di = np.array([(pd.Timestamp(d) - day0).days for d in dates])
        n = di[-1] + 1
        _, H, W = nd.shape
        flat_nd = nd.reshape(len(dates), -1); flat_cl = (clr == 1).reshape(len(dates), -1)
        out = np.full(flat_nd.shape, -32768, dtype="int16")
        res = detect_resources()
        from safe_resources import safe_memory_bytes          # size by the pod's memory REQUEST, not the free RAM
        chunk = max(10000, int(safe_memory_bytes(0.3) / max(res.cpus, 1) / (n * 8 * 5)))

        def run(s):
            e = min(s + chunk, flat_nd.shape[1])
            cl = flat_cl[:, s:e]
            y = np.zeros((n, e - s)); w = np.zeros((n, e - s))
            y[di] = np.where(cl, flat_nd[:, s:e] / 10000.0, 0); w[di] = cl
            z = whittaker_daily(y, w)[di]
            ok = cl.sum(0) >= 4                                # need a few clear points for a meaningful fit
            out[:, s:e] = np.where(ok, np.round(np.clip(z, -1, 1) * 10000), -32768).astype("int16")

        with ThreadPoolExecutor(res.cpus) as pool:
            list(pool.map(run, range(0, flat_nd.shape[1], chunk)))
        write("S2_NDVI_WS", out.reshape(nd.shape), dates, prof.copy(), -32768,
              {"variable": "Whittaker NDVI x 10000", "lambda": str(LAMBDA), "d": str(ORDER), "grid": "daily, weights = S2_CLEAR"})
        print(f"  Whittaker stack {time.time() - t0:.0f} s ({chunk} px per chunk)", flush=True)

    # fixed chip stretch from clear AOI pixels (sampled)
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        inside = r.read(2).astype(bool).ravel()
    rng = np.random.default_rng(0)
    stats = {}
    for b in ["B5", "B3", "B2"]:
        a, _, _ = read(f"S2_{b}")
        a = a.reshape(len(dates), -1)[:, inside]; c = (clr.reshape(len(dates), -1)[:, inside] == 1)
        v = a[c]; v = rng.choice(v, min(v.size, 2_000_000), replace=False)
        stats[b] = [float(np.percentile(v, 2)), float(np.percentile(v, 98))]
    (ST / "s2_stretch.json").write_text(json.dumps(stats, indent=1))
    print("  stretch", stats, flush=True)


# ---------------------------------------------------------------- rain rasters
def rain_layers():
    import ee, rain
    rain.init_ee(rain.CFG, high_volume=True)
    aoi = rain.aoi_geom()
    b = aoi.bounds(1).buffer(12000).bounds(1).coordinates().getInfo()[0]
    res = 0.1
    x0 = np.floor(min(p[0] for p in b) / res) * res; y1 = np.ceil(max(p[1] for p in b) / res) * res
    w = int(np.ceil((max(p[0] for p in b) - x0) / res)); h = int(np.ceil((y1 - min(p[1] for p in b)) / res))
    grid = {"dimensions": {"width": w, "height": h}, "crsCode": "EPSG:4326",
            "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -res, "translateY": y1}}

    def pull(images, names, out, tags):
        img = ee.Image.cat(images).rename(names)
        arrs = []
        for i in range(0, len(names), 100):                           # keep each request small
            a = ee.data.computePixels({"expression": img.select(names[i:i + 100]), "fileFormat": "NUMPY_NDARRAY", "grid": grid})
            arrs += [a[n] for n in names[i:i + 100]]
        stack = np.stack(arrs).astype("float32")
        with rasterio.open(PROJ / "data/rain" / out, "w", driver="GTiff", width=w, height=h, count=len(names), dtype="float32",
                           crs="EPSG:4326", transform=rasterio.transform.from_origin(x0, y1, res, res), compress="deflate") as dst:
            dst.write(stack)
            for k, n in enumerate(names, 1):
                dst.set_band_description(k, n)
            dst.update_tags(**tags)
        print(f"  {out}: {len(names)} bands, {w}x{h} cells", flush=True)

    def total(cid, band, f, t0, t1):
        """Sum of the product between two dates (mm); -1 where the product has no image yet."""
        col = ee.ImageCollection(cid).filterDate(t0, t1).select(band)
        return ee.Image(ee.Algorithms.If(col.size().gt(0), col.sum().multiply(f), ee.Image.constant(-1))).rename("r").unmask(-1)

    # IMERG daily (UTC days)
    days = pd.date_range(rain.START, pd.Timestamp(rain.END) - pd.Timedelta(days=1), freq="D")
    cid, band, f, _ = rain.HOURLY["imerg"]
    imgs = [total(cid, band, f, d.strftime("%Y-%m-%d"), (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")) for d in days]
    pull(imgs, [d.strftime("%Y%m%d") for d in days], "imerg_daily.tif", {"units": "mm/day (UTC day)", "source": cid})

    # 24 h before every S1 acquisition
    man = pd.read_csv(PROJ / "data/manifest/download_manifest.csv")
    s1 = man[man.sensor == "s1"].copy()
    s1["t"] = s1.scenes.str.split(";").str[0].str.split("_").str[4].map(
        lambda t: datetime.strptime(t, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).isoformat())
    names = [f"{r.track}_{r.date}" for r in s1.itertuples()]
    for prod in ["imerg", "gsmap"]:
        cid, band, f, _ = rain.HOURLY[prod]
        imgs = [total(cid, band, f, ee.Date(t).advance(-24, "hour"), ee.Date(t)) for t in s1.t]
        pull(imgs, names, f"s1_rain24h_{prod}.tif", {"units": "mm in the 24 h before the S1 acquisition", "source": cid})

    t = rain.s1_table(aoi)
    t.round(2).to_csv(PROJ / "data/rain/rain_s1_acquisitions.csv", index=False)
    print(f"  rain_s1_acquisitions.csv: {len(t)} rows", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    part = sys.argv[1] if len(sys.argv) > 1 else "all"           # all | s2 | rain
    if part in ("all", "s2"):
        s2_layers()
    if part in ("all", "rain"):
        rain_layers()
    print(f"done in {(time.time() - t0) / 60:.1f} min")

"""Look at any pixel: its time-series curves and image chips around it.

A pixel can be given as pixel_id (band 1 of data/grid/index.tif), as x/y in EPSG:32648, or as lon/lat.
Uses the time stacks in data/stacks/ (run scripts/build_stacks.py first).

Python:
    from pixel_tools import curves, plot_curves, chip
    df = curves(lon=103.31, lat=14.45, radius=2)        # radius 2 = 5 x 5 window mean (S1 averaged in linear power)
    plot_curves(lon=103.31, lat=14.45, out="pixel.png")
    chip(lon=103.31, lat=14.45, date="20260115", size=128, bands="B8,B4,B3", out="chip.png")
CLI:
    python scripts/pixel_tools.py curve --lon 103.31 --lat 14.45 --radius 2 --out pixel.png
    python scripts/pixel_tools.py chip  --pixel-id 1234567 --date 20260115 --bands B5,B3,B2
"""
from __future__ import annotations

from pathlib import Path
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

PROJ = Path(__file__).resolve().parents[1]
STACKS = PROJ / "data/stacks"
INDEX = PROJ / "data/grid/index.tif"
CLOUDY_SCL = {3, 8, 9, 10}                     # cloud shadow, cloud medium/high probability, thin cirrus


def locate(pixel_id=None, x=None, y=None, lon=None, lat=None):
    """-> (row, col, pixel_id, in_aoi)"""
    with rasterio.open(INDEX) as r:
        W, H = r.width, r.height
        if pixel_id is not None:
            row, col = divmod(int(pixel_id), W)
        else:
            if lon is not None:
                from pyproj import Transformer
                x, y = Transformer.from_crs(4326, r.crs, always_xy=True).transform(lon, lat)
            row, col = r.index(x, y)
        if not (0 <= row < H and 0 <= col < W):
            raise ValueError("point is outside the grid")
        inside = int(r.read(2, window=Window(col, row, 1, 1))[0, 0])
    return row, col, row * W + col, inside


def _read(name, row, col, radius):
    with rasterio.open(STACKS / f"{name}.tif") as r:
        r0, c0 = max(0, row - radius), max(0, col - radius)
        win = Window(c0, r0, min(r.width, col + radius + 1) - c0, min(r.height, row + radius + 1) - r0)
        a = r.read(window=win).astype("float64")
        nod = r.nodata
        if nod is not None and not np.isnan(nod):
            a[a == nod] = np.nan
        return a.reshape(a.shape[0], -1), [d for d in r.descriptions]


def curves(pixel_id=None, x=None, y=None, lon=None, lat=None, radius=0):
    """Long table: date, sensor, track, variable, value (window mean if radius > 0)."""
    row, col, pid, inside = locate(pixel_id, x, y, lon, lat)
    meta = pd.read_csv(STACKS / "stacks.csv")
    out = []
    for m in meta.itertuples():
        a, dates = _read(m.name, row, col, radius)
        if m.sensor == "s1" and m.variable in ("VV", "VH"):
            with np.errstate(all="ignore"):
                v = 10 * np.log10(np.nanmean(10 ** (a / 10), axis=1))        # average in linear power, back to dB
        elif m.variable in ("SCL", "QA60", "MSK_CLASSI_OPAQUE", "MSK_CLASSI_CIRRUS"):
            v = np.array([pd.Series(r).mode().iloc[0] if np.isfinite(r).any() else np.nan for r in a])  # categorical
        else:
            with np.errstate(all="ignore"):
                v = np.nanmean(a, axis=1)
            if m.variable.startswith("B"):
                v = v / 10000.0
        out.append(pd.DataFrame({"date": pd.to_datetime(dates), "sensor": m.sensor, "track": m.track,
                                 "variable": m.variable, "value": v}))
    df = pd.concat(out, ignore_index=True)
    df.attrs.update(row=row, col=col, pixel_id=pid, in_aoi=inside, radius=radius)
    return df


def s2_table(df):
    """Wide S2 table with NDVI and a clear flag (Cloud Score+ cs_cdf >= 60 and SCL not cloud/shadow)."""
    s2 = df[df.sensor == "s2"].pivot_table(index="date", columns="variable", values="value")
    s2["NDVI"] = (s2.B8 - s2.B4) / (s2.B8 + s2.B4)
    s2["clear"] = (s2.get("cs_cdf", 100) >= 60) & ~s2.get("SCL", pd.Series(4, index=s2.index)).isin(CLOUDY_SCL)
    return s2


def plot_curves(pixel_id=None, x=None, y=None, lon=None, lat=None, radius=0, out="pixel_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = curves(pixel_id, x, y, lon, lat, radius)
    a = df.attrs
    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for (track, var), g in df[(df.sensor == "s1") & df.variable.isin(["VV", "VH"])].groupby(["track", "variable"]):
        ax[0 if var == "VH" else 1].plot(g.date, g.value, marker=".", label=f"{track} {var}")
    ax[0].set_ylabel("VH (dB)"); ax[1].set_ylabel("VV (dB)")
    s2 = s2_table(df)
    ax[2].plot(s2.index, s2.NDVI, color="lightgray", lw=1, label="NDVI all dates")
    ax[2].scatter(s2.index[s2.clear], s2.NDVI[s2.clear], color="green", s=25, zorder=3, label="NDVI clear")
    ax[2].scatter(s2.index[~s2.clear], s2.NDVI[~s2.clear], color="gray", marker="x", s=15, label="cloudy")
    ax[2].set_ylabel("NDVI"); ax[2].set_ylim(-0.2, 1)
    for x_ in ax:
        x_.grid(alpha=0.3); x_.legend(fontsize=8, loc="best")
    fig.suptitle(f"pixel_id {a['pixel_id']} (row {a['row']}, col {a['col']}), in_aoi={a['in_aoi']}, "
                 f"window {2 * radius + 1}x{2 * radius + 1}")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def chip(pixel_id=None, x=None, y=None, lon=None, lat=None, date=None, size=128, bands="B4,B3,B2",
         out=None, stretch=(2, 98)):
    """PNG chip (size x size px) around the pixel for one date. bands: 3 S2 bands (e.g. B8,B4,B3 or B5,B3,B2)
    or 'S1_RO164_DES' / 'S1_RO099_ASC' for a VV,VH,VV/VH false colour. The centre pixel is marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row, col, pid, _ = locate(pixel_id, x, y, lon, lat)
    h = size // 2
    if bands.startswith("S1_"):
        names = [f"{bands}_VV", f"{bands}_VH"]
    else:
        names = [f"S2_{b}" for b in bands.split(",")]
    layers = []
    for n in names:
        with rasterio.open(STACKS / f"{n}.tif") as r:
            dates = list(r.descriptions)
            k = dates.index(date) + 1 if date in dates else 1 + int(np.argmin([abs(pd.Timestamp(d) - pd.Timestamp(date)).days for d in dates]))
            win = Window(max(0, col - h), max(0, row - h), size, size)
            layers.append(r.read(k, window=win, boundless=True, fill_value=0).astype("float64"))
            used = dates[k - 1]
    if bands.startswith("S1_"):
        layers = [layers[0], layers[1], layers[0] - layers[1]]
    rgb = np.dstack([(l - np.nanpercentile(l, stretch[0])) / (np.nanpercentile(l, stretch[1]) - np.nanpercentile(l, stretch[0]) + 1e-9)
                     for l in layers]).clip(0, 1)
    out = out or f"chip_{pid}_{used}_{bands.replace(',', '')}.png"
    plt.figure(figsize=(5, 5)); plt.imshow(rgb); plt.scatter([min(h, col)], [min(h, row)], s=60, facecolors="none", edgecolors="red")
    plt.title(f"{bands} {used}  pixel {pid}"); plt.axis("off"); plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["curve", "chip"])
    ap.add_argument("--pixel-id", type=int); ap.add_argument("--x", type=float); ap.add_argument("--y", type=float)
    ap.add_argument("--lon", type=float); ap.add_argument("--lat", type=float)
    ap.add_argument("--radius", type=int, default=0); ap.add_argument("--date"); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bands", default="B4,B3,B2"); ap.add_argument("--out")
    a = ap.parse_args()
    kw = dict(pixel_id=a.pixel_id, x=a.x, y=a.y, lon=a.lon, lat=a.lat)
    if a.cmd == "curve":
        print(plot_curves(**kw, radius=a.radius, out=a.out or "pixel_curves.png"))
        print(s2_table(curves(**kw, radius=a.radius)).round(3).to_string())
    else:
        print(chip(**kw, date=a.date, size=a.size, bands=a.bands, out=a.out))

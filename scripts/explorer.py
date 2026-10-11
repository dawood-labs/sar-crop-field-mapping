"""Fast pixel explorer: NDVI curves (raw, cloud-masked, Whittaker), Sentinel-1 VV/VH with rain, and S2 chips.

Everything heavy is precomputed (scripts/build_stacks.py + scripts/precompute.py); here we only read one pixel
(or one small window) from tiled time stacks, so a pixel loads in about a second.

    import explorer as ex
    px = ex.load_pixel(pixel_id=2400000)                 # or lon=..., lat=...  or x=..., y=... (EPSG:32648)
    ex.plot_ndvi(px, start="2025-09-01", end="2026-10-08", backend="plotly", lam=1.0, d=2)
    ex.plot_s1(px, start=..., end=..., backend="matplotlib", rain="imerg")
    ex.plot_chips(px, start=..., end=..., size=64, bands=("B5", "B3", "B2"), stretch="fixed")
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

PROJ = Path(__file__).resolve().parents[1]
ST, RAIN = PROJ / "data/stacks", PROJ / "data/rain"
TRACK_COLORS = {"RO164_DES": "#1f77b4", "RO099_ASC": "#d62728"}


# ---------------------------------------------------------------- low-level reads (datasets kept open)
@lru_cache(maxsize=None)
def _ds(path):
    return rasterio.open(path)


def _dates(path):
    return [str(d) for d in _ds(path).descriptions]


def _pixel(name, row, col, radius=0):
    """All dates of one stack at a pixel (or the mean of a (2r+1)^2 window) as float with NaN for no data."""
    r = _ds(ST / f"{name}.tif")
    win = Window(col - radius, row - radius, 2 * radius + 1, 2 * radius + 1)
    a = r.read(window=win, boundless=True, fill_value=r.nodata if r.nodata is not None else 0).astype("float64")
    if r.nodata is not None and not np.isnan(r.nodata):
        a[a == r.nodata] = np.nan
    return a.reshape(a.shape[0], -1)


@lru_cache(maxsize=1)
def _grid():
    with rasterio.open(PROJ / "data/grid/index.tif") as r:
        return r.width, r.height, r.transform, r.crs


def locate(pixel_id=None, x=None, y=None, lon=None, lat=None):
    """-> row, col, pixel_id, x, y (pixel centre in EPSG:32648)."""
    W, H, tr, crs = _grid()
    if pixel_id is not None:
        row, col = divmod(int(pixel_id), W)
    else:
        if lon is not None:
            from pyproj import Transformer
            x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
        col, row = int((x - tr.c) // tr.a), int((tr.f - y) // -tr.e)
    if not (0 <= row < H and 0 <= col < W):
        raise ValueError(f"pixel outside the grid (rows 0-{H - 1}, cols 0-{W - 1})")
    xc, yc = tr.c + (col + 0.5) * tr.a, tr.f + (row + 0.5) * tr.e
    return row, col, row * W + col, xc, yc


def _rain_at(path, lon, lat):
    r = _ds(path)
    rr, cc = r.index(lon, lat)
    a = r.read(window=Window(cc, rr, 1, 1)).ravel().astype("float64")
    a[a < 0] = np.nan                                           # -1 = product had no data yet
    return pd.Series(a, index=_dates(path))


# ---------------------------------------------------------------- one pixel, everything
def load_pixel(pixel_id=None, x=None, y=None, lon=None, lat=None, radius=0):
    """Read every time series for a pixel. radius > 0 averages a window (S1 in linear power)."""
    row, col, pid, xc, yc = locate(pixel_id, x, y, lon, lat)
    from pyproj import Transformer
    lon_c, lat_c = Transformer.from_crs(_grid()[3], 4326, always_xy=True).transform(xc, yc)
    inside = int(_ds(PROJ / "data/grid/index.tif").read(2, window=Window(col, row, 1, 1))[0, 0])

    # Sentinel-2
    dates = pd.to_datetime(_dates(ST / "S2_NDVI.tif"))
    nd = np.nanmean(_pixel("S2_NDVI", row, col, radius), 1) / 1e4
    clear = _pixel("S2_CLEAR", row, col, radius)
    clear = np.nanmean(clear, 1) >= 0.5 if radius else clear[:, 0] == 1
    ws = np.nanmean(_pixel("S2_NDVI_WS", row, col, radius), 1) / 1e4
    cs = np.nanmean(_pixel("S2_cs_cdf", row, col, radius), 1)
    scl = _pixel("S2_SCL", row, col, 0)[:, 0]
    s2 = pd.DataFrame({"date": dates, "ndvi": nd, "clear": clear, "ndvi_ws_precomputed": ws, "cs_cdf": cs, "scl": scl})

    # Sentinel-1 (all tracks found in the stacks)
    s1 = []
    for track in sorted({p.name.split("_")[1] + "_" + p.name.split("_")[2] for p in ST.glob("S1_RO*_VV.tif")}):
        d = pd.to_datetime(_dates(ST / f"S1_{track}_VV.tif"))
        v = {}
        for var in ["VV", "VH"]:
            a = _pixel(f"S1_{track}_{var}", row, col, radius)
            with np.errstate(all="ignore"):
                v[var] = 10 * np.log10(np.nanmean(10 ** (a / 10), 1))      # window mean in linear power
        v["angle"] = np.nanmean(_pixel(f"S1_{track}_angle", row, col, radius), 1)
        s1.append(pd.DataFrame({"date": d, "track": track, **v}))
    s1 = pd.concat(s1, ignore_index=True)
    s1["VH_minus_VV"] = s1.VH - s1.VV
    for prod in ["imerg", "gsmap"]:                                       # rain in the 24 h before each acquisition
        rr = _rain_at(RAIN / f"s1_rain24h_{prod}.tif", lon_c, lat_c)
        s1[f"rain24h_{prod}"] = [rr.get(f"{t}_{d:%Y%m%d}", np.nan) for t, d in zip(s1.track, s1.date)]

    # daily rain at the pixel: CHIRPS where it exists, IMERG after
    ch = _rain_at(RAIN / "chirps_daily.tif", lon_c, lat_c); ch.index = pd.to_datetime(ch.index)
    im = _rain_at(RAIN / "imerg_daily.tif", lon_c, lat_c); im.index = pd.to_datetime(im.index)
    daily = pd.DataFrame({"chirps": ch, "imerg": im}).sort_index()
    daily["best"] = daily.chirps.fillna(daily.imerg)
    return {"row": row, "col": col, "pixel_id": pid, "x": xc, "y": yc, "lon": lon_c, "lat": lat_c, "in_aoi": inside,
            "radius": radius, "s2": s2, "s1": s1, "rain_daily": daily}


# ---------------------------------------------------------------- Whittaker for one pixel (any lambda / d)
def whittaker(dates, values, weights, lam=1.0, d=2):
    """Weighted Whittaker smoother on a DAILY grid (Eilers 2003). Returns (daily dates, smoothed values)."""
    dates = pd.DatetimeIndex(dates)
    day0 = dates.min()
    idx = np.asarray((dates - day0).days)
    n = idx.max() + 1
    y = np.zeros(n); w = np.zeros(n)
    ok = np.isfinite(values) & (weights > 0)
    y[idx[ok]] = values[ok]; w[idx[ok]] = weights[ok]
    if ok.sum() <= d:
        return pd.date_range(day0, periods=n), np.full(n, np.nan)
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    D = sparse.eye(n, format="csc")
    for _ in range(d):
        D = D[1:] - D[:-1]
    z = spsolve(sparse.diags(w) + lam * (D.T @ D), w * y)
    return pd.date_range(day0, periods=n), z


def _clip(df, start, end, col="date"):
    s, e = pd.Timestamp(start) if start else df[col].min(), pd.Timestamp(end) if end else df[col].max()
    return df[(df[col] >= s) & (df[col] <= e)]


def _title(px):
    w = 2 * px["radius"] + 1
    return (f"pixel_id {px['pixel_id']}  (row {px['row']}, col {px['col']}, lon {px['lon']:.5f}, lat {px['lat']:.5f}, "
            f"in AOI={px['in_aoi']}, window {w}x{w})")


# ---------------------------------------------------------------- plot 1: NDVI
def plot_ndvi(px, start=None, end=None, backend="matplotlib", lam=1.0, d=2, figsize=(14, 5)):
    s2 = px["s2"]
    dd, z = whittaker(s2.date, s2.ndvi.values, s2.clear.astype(float).values, lam, d)   # fit on the full period
    ws = _clip(pd.DataFrame({"date": dd, "ws": z}), start, end)
    s2 = _clip(s2, start, end)
    clr = s2[s2.clear]
    if backend == "plotly":
        import plotly.graph_objects as go
        f = go.Figure()
        f.add_scatter(x=s2.date, y=s2.ndvi, mode="lines+markers", name="NDVI raw (all dates)",
                      line=dict(color="lightgray"), marker=dict(color=np.where(s2.clear, "gray", "lightgray"), size=5),
                      customdata=np.c_[s2.cs_cdf, s2.scl], hovertemplate="%{x|%Y-%m-%d}<br>NDVI %{y:.3f}<br>cs_cdf %{customdata[0]:.0f}  SCL %{customdata[1]:.0f}")
        f.add_scatter(x=clr.date, y=clr.ndvi, mode="lines+markers", name=f"NDVI cloud-masked ({len(clr)} clear)",
                      line=dict(color="green"), marker=dict(size=7))
        f.add_scatter(x=ws.date, y=ws.ws, mode="lines", name=f"Whittaker (lambda={lam}, d={d})", line=dict(color="royalblue", width=3))
        f.update_layout(title=_title(px), yaxis_title="NDVI", yaxis_range=[-0.2, 1], height=450, hovermode="x unified")
        f.show()
        return f
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s2.date, s2.ndvi, color="lightgray", lw=1, marker="x", ms=4, label="NDVI raw (all dates)")
    ax.plot(clr.date, clr.ndvi, color="green", marker="o", ms=5, lw=1.5, label=f"NDVI cloud-masked ({len(clr)} clear)")
    ax.plot(ws.date, ws.ws, color="royalblue", lw=2.5, label=f"Whittaker (lambda={lam}, d={d})")
    ax.set_ylim(-0.2, 1); ax.set_ylabel("NDVI"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax.set_title(_title(px), fontsize=10); fig.tight_layout(); plt.show()
    return fig


# ---------------------------------------------------------------- plot 2: S1 VV / VH + rain
def plot_s1(px, start=None, end=None, backend="matplotlib", rain="imerg", show_daily_rain=True, figsize=(14, 8)):
    """VH (top) and VV (bottom) of every track; bars = rain in the 24 h before each acquisition at this pixel
    (rain='imerg' or 'gsmap'); thin gray bars = daily rain (CHIRPS to its last date, IMERG after)."""
    s1 = _clip(px["s1"], start, end)
    daily = px["rain_daily"]
    daily = daily[(daily.index >= (pd.Timestamp(start) if start else daily.index.min())) &
                  (daily.index <= (pd.Timestamp(end) if end else daily.index.max()))]
    rc = f"rain24h_{rain}"
    if backend == "plotly":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        f = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                          specs=[[{"secondary_y": True}], [{"secondary_y": True}]], subplot_titles=["VH", "VV"])
        for i, var in enumerate(["VH", "VV"], 1):
            if show_daily_rain:
                f.add_bar(x=daily.index, y=daily.best, name="daily rain (mm)", marker_color="lightgray", opacity=0.5,
                          showlegend=i == 1, secondary_y=True, row=i, col=1)
            for t, g in s1.groupby("track"):
                c = TRACK_COLORS.get(t, None)
                f.add_bar(x=g.date, y=g[rc], name=f"{t} rain 24h before ({rain})", marker_color=c, opacity=0.35,
                          width=86400000 * 1.5, showlegend=i == 1, secondary_y=True, row=i, col=1)
                f.add_scatter(x=g.date, y=g[var], mode="lines+markers", name=f"{t} {var}", line=dict(color=c),
                              customdata=np.c_[g[rc], g.angle], secondary_y=False, row=i, col=1,
                              hovertemplate="%{x|%Y-%m-%d} " + var + " %{y:.2f} dB<br>rain 24h %{customdata[0]:.1f} mm, angle %{customdata[1]:.1f}")
            f.update_yaxes(title_text=f"{var} (dB)", secondary_y=False, row=i, col=1)
            f.update_yaxes(title_text="rain (mm)", secondary_y=True, row=i, col=1, showgrid=False)
        f.update_layout(title=_title(px), height=700, barmode="overlay", hovermode="x unified")
        f.show()
        return f
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, var in zip(axes, ["VH", "VV"]):
        ax2 = ax.twinx()
        if show_daily_rain:
            ax2.bar(daily.index, daily.best, width=1, color="lightgray", alpha=0.6, label="daily rain")
        for t, g in s1.groupby("track"):
            c = TRACK_COLORS.get(t, None)
            ax2.bar(g.date, g[rc], width=2, color=c, alpha=0.3, label=f"{t} rain 24h before ({rain})")
            ax.plot(g.date, g[var], marker="o", ms=4, color=c, label=f"{t} {var}")
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_ylabel(f"{var} (dB)"); ax2.set_ylabel("rain (mm)"); ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", ncol=2)
    axes[0].set_title(_title(px), fontsize=10); fig.tight_layout(); plt.show()
    return fig


# ---------------------------------------------------------------- S2 chips on clear dates
@lru_cache(maxsize=1)
def _stretch():
    return json.loads((ST / "s2_stretch.json").read_text())


def plot_chips(px, start=None, end=None, size=64, bands=("B5", "B3", "B2"), stretch="fixed", ncols=6,
               backend="matplotlib", max_chips=60):
    """RGB chips (size x size pixels, 10 m) around the pixel for every date where THIS pixel is cloud-free.
    stretch='fixed' uses the AOI-wide 2-98 % of each band (dates comparable); 'chip' stretches each chip."""
    s2 = _clip(px["s2"], start, end)
    s2 = s2[s2.clear]
    if s2.empty:
        print("no cloud-free date for this pixel in the chosen range"); return None
    s2 = s2.head(max_chips)
    all_dates = list(px["s2"].date)
    idx = [all_dates.index(d) + 1 for d in s2.date]
    h = size // 2
    win = Window(px["col"] - h, px["row"] - h, size, size)
    layers = [_ds(ST / f"S2_{b}.tif").read(idx, window=win, boundless=True, fill_value=0).astype("float32") for b in bands]
    chips = []
    for k in range(len(idx)):
        rgb = []
        for b, L in zip(bands, layers):
            a = L[k]
            if stretch == "fixed" and b in _stretch():
                lo, hi = _stretch()[b]
            else:                                   # per-chip stretch (also used for bands without a fixed stretch)
                lo, hi = np.nanpercentile(a[a > 0], [2, 98]) if (a > 0).any() else (0, 1)
            rgb.append(np.clip((a - lo) / (hi - lo + 1e-9), 0, 1))
        chips.append(np.dstack(rgb))
    n = len(chips); nr = int(np.ceil(n / ncols))
    lab = [f"{d:%Y-%m-%d}" for d in s2.date]
    if backend == "plotly":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        f = make_subplots(rows=nr, cols=ncols, subplot_titles=lab, horizontal_spacing=0.01, vertical_spacing=0.04)
        for k, c in enumerate(chips):
            f.add_trace(go.Image(z=(c * 255).astype("uint8")), row=k // ncols + 1, col=k % ncols + 1)
            f.add_shape(type="rect", x0=h - 1.5, x1=h + 0.5, y0=h - 1.5, y1=h + 0.5, line=dict(color="red", width=2),
                        row=k // ncols + 1, col=k % ncols + 1)
        f.update_xaxes(visible=False); f.update_yaxes(visible=False)
        f.update_layout(title=f"S2 {'/'.join(bands)} cloud-free chips, {size}x{size} px - " + _title(px),
                        height=220 * nr + 80, margin=dict(l=10, r=10, t=80, b=10))
        f.show()
        return f
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nr, ncols, figsize=(2.6 * ncols, 3.0 * nr + 0.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, c in enumerate(chips):
        ax = axes[k // ncols, k % ncols]
        ax.imshow(c); ax.add_patch(plt.Rectangle((h - 1.5, h - 1.5), 2, 2, fill=False, ec="red", lw=1.5))
        ax.set_title(lab[k], fontsize=9)
    fig.suptitle(f"S2 {'/'.join(bands)} cloud-free chips ({n}), {size}x{size} px - " + _title(px), fontsize=10)
    fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.97)); plt.show()
    return fig

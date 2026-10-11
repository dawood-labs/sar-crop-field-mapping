"""Independent audit of rice model v1: sample 150 pixels by predicted probability and draw BLIND review sheets.

Sampling (data/model/rice_v1/rice_prob.tif): rice side 50 (>=90), 30 (70-89), 20 (50-69); non-rice side 20 (<=10),
15 (11-30), 15 (31-49). Pixels are >= 300 m apart and >= 300 m from every GCP and every training sample, so the audit
is independent of training. IDs A001..A150 are shuffled so a batch says nothing about the model's answer.

Outputs (data/model/rice_v1/audit/):
  key/audit_key.csv    audit_id, row, col, lon, lat, model prob, model class, bin   (NOT shown to the analysts)
  sheets/A###.png      blind sheet: S1 VH/VV/VH-VV both tracks (3x3, Sep 2025 - Oct 2026) with rain and the median
                       curves of verified rice / tree / sugarcane GCPs; cloud-free NDVI with the same class medians;
                       10 cloud-free B5/B3/B2 chips; 2 context chips (1.28 km) B5/B3/B2 and B8/B4/B3; DEM, slope
  points.csv           audit_id, lon, lat (for the analysts / QGIS)
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")   # MB per process; default is 5 % of host RAM (see safe_resources.py)

import json, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import explorer as ex  # noqa: E402

# usage: audit_sheets.py [model_version]   (rice_v1 default; rice_v2: 100 rice + 100 non-rice, also avoids v1 audit pixels)
VER = sys.argv[1] if len(sys.argv) > 1 else "rice_v1"
M = PROJ / "data/model" / VER
A = M / "audit"
BINS_V2 = [("rice_high", 90, 100, 50), ("rice_med", 70, 89, 30), ("rice_low", 50, 69, 20),
           ("non_low", 0, 10, 40), ("non_med", 11, 30, 30), ("non_near", 31, 49, 30)]
BINS_V3 = [("rice_high", 90, 100, 25), ("rice_med", 70, 89, 15), ("rice_low", 50, 69, 10),
           ("non_low", 0, 10, 20), ("non_med", 11, 30, 15), ("non_near", 31, 49, 15)]
BINS = [("rice_high", 90, 100, 50), ("rice_med", 70, 89, 30), ("rice_low", 50, 69, 20),
        ("non_low", 0, 10, 20), ("non_med", 11, 30, 15), ("non_near", 31, 49, 15)]
TC = ex.TRACK_COLORS
STRETCH = json.loads((PROJ / "data/stacks/s2_stretch.json").read_text())


def sample():
    from scipy.spatial import cKDTree
    rng = np.random.default_rng({"rice_v1": 42, "rice_v2": 4242}.get(VER, 424242))
    bins = {"rice_v1": BINS, "rice_v2": BINS_V2}.get(VER, BINS_V3)
    prob = rasterio.open(M / "rice_prob.tif").read(1)
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")
    import geopandas as gpd
    s = gpd.read_file(M / "samples.gpkg")
    avoid = [g[["row", "col"]].to_numpy(), s[["row", "col"]].to_numpy()]
    for k in (PROJ / "data/model").glob("rice_v*/audit/key/audit_key.csv"):   # never re-use a pixel from an earlier audit
        if k.parts[-4] != VER:
            avoid.append(pd.read_csv(k)[["row", "col"]].to_numpy())
    avoid = np.vstack(avoid)
    tav = cKDTree(avoid)
    pts, rows = [], []
    for name, lo, hi, n in bins:
        rr, cc = np.nonzero((prob >= lo) & (prob <= hi))
        order = rng.permutation(len(rr))
        got = 0
        for i in order:
            p = (rr[i], cc[i])
            if tav.query(p)[0] < 30 or (pts and cKDTree(pts).query(p)[0] < 30):
                continue
            pts.append(p); rows.append(dict(row=int(p[0]), col=int(p[1]), model_prob=int(prob[p]), bin=name)); got += 1
            if got >= n:
                break
    d = pd.DataFrame(rows)
    d["model_class"] = np.where(d.model_prob >= 50, "Rice", "Non-rice")
    d = d.sample(frac=1, random_state=7).reset_index(drop=True)
    d.insert(0, "audit_id", [f"A{i + 1:03d}" for i in range(len(d))])
    W, H, tr, crs = ex._grid()
    from pyproj import Transformer
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    xs, ys = tr.c + (d.col + 0.5) * 10, tr.f - (d.row + 0.5) * 10
    d["lon"], d["lat"] = t.transform(xs.values, ys.values)
    return d


def templates():
    """Median curves of verified GCP classes: NDVI per month (cloud-free) and VH per S1 date per track (3x3)."""
    S = pickle.load(open(PROJ / "data/gcps/eda/gcp_series.pkl", "rb"))
    v = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv").set_index("gcp_id")
    grp = {"rice": ["Rice"], "tree": ["Tree"], "sugarcane": ["Sugarcane"]}
    out = {}
    for k, cl in grp.items():
        ids = [i for i in v.index if v.loc[i, "usable_for_training"] and v.loc[i, "class_2026"] in cl]
        nd = pd.concat([S[i]["s2"][S[i]["s2"].clear].set_index("date").ndvi for i in ids])
        ndm = nd.groupby(nd.index.to_period("M")).median()
        vh = {}
        for t in TC:
            vv = pd.concat([S[i]["s1_3x3"][S[i]["s1_3x3"].track == t].set_index("date").VH for i in ids], axis=1)
            vh[t] = vv.median(axis=1)
        out[k] = (pd.Series(ndm.values, index=ndm.index.to_timestamp() + pd.Timedelta(days=14)), vh, len(ids))
    return out


def chips(row, col, dates, bands, size, stretch="fixed"):
    allds = list(pd.to_datetime(ex._dates(PROJ / "data/stacks/S2_NDVI.tif")))
    idx = [allds.index(d) + 1 for d in dates]
    h = size // 2
    win = Window(col - h, row - h, size, size)
    lay = [ex._ds(PROJ / f"data/stacks/S2_{b}.tif").read(idx, window=win, boundless=True, fill_value=0).astype("float32") for b in bands]
    out = []
    for k in range(len(idx)):
        ch = []
        for b, L in zip(bands, lay):
            a = L[k]
            lo, hi = STRETCH[b] if (stretch == "fixed" and b in STRETCH) else np.percentile(a[a > 0], [2, 98]) if (a > 0).any() else (0, 1)
            ch.append(np.clip((a - lo) / (hi - lo + 1e-9), 0, 1))
        out.append(np.dstack(ch))
    return out


def sheet(r, T, dem):
    px1 = ex.load_pixel(pixel_id=r.row * ex._grid()[0] + r.col, radius=0)
    px3 = ex.load_pixel(pixel_id=r.row * ex._grid()[0] + r.col, radius=1)
    s2, s1, rain = px1["s2"], px3["s1"], px1["rain_daily"]
    clr = s2[s2.clear]
    fig = plt.figure(figsize=(17, 16))
    gs = fig.add_gridspec(6, 10, height_ratios=[0.25, 1.15, 1.0, 1.0, 0.8, 1.9], hspace=0.5)
    a0 = fig.add_subplot(gs[0, :]); a0.axis("off")
    z = dem.read(window=Window(r.col, r.row, 1, 1)).ravel()
    a0.text(0, 0.9, f"{r.audit_id}    lon {r.lon:.5f}  lat {r.lat:.5f}    elevation {z[0]:.0f} m, slope {z[1]:.1f} deg    "
            f"cloud-free S2 dates: {len(clr)}", fontsize=13, weight="bold", va="top")
    a0.text(0, 0.1, "dashed = median of VERIFIED ground points: green rice, brown tree crops (rubber/eucalyptus), purple sugarcane",
            fontsize=9, va="top")
    tcol = {"rice": "green", "tree": "saddlebrown", "sugarcane": "purple"}
    a1 = fig.add_subplot(gs[1, :])
    a1.plot(clr.date, clr.ndvi, "o-", color="black", ms=5, lw=1.4, label="this pixel, cloud-free NDVI")
    for k, (nd, _, n) in T.items():
        a1.plot(nd.index, nd.values, "--", color=tcol[k], lw=1.3, label=f"{k} (n={n})")
    a1.set_ylim(-0.1, 1); a1.set_ylabel("NDVI"); a1.grid(alpha=0.3); a1.legend(fontsize=8, ncol=4, loc="upper right")
    a1.axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-30"), color="orange", alpha=0.07)
    a1.axvspan(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-08-31"), color="blue", alpha=0.05)
    for i, t in enumerate(TC):
        a = fig.add_subplot(gs[2 + i, :], sharex=a1); a2 = a.twinx()
        g = s1[s1.track == t]
        a2.bar(rain.index, rain.best, width=1, color="lightgray", alpha=0.6)
        a.plot(g.date, g.VH, "o-", ms=3.5, lw=1.4, color=TC[t], label=f"VH {t} (3x3)")
        a.plot(g.date, g.VV, "-", lw=0.9, color=TC[t], alpha=0.45, label="VV")
        for k, (_, vh, _) in T.items():
            a.plot(vh[t].index, vh[t].values, "--", color=tcol[k], lw=1.1)
        a.axhline(-20, color="k", ls=":", lw=1)
        a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
        a.set_ylabel("dB"); a2.set_ylabel("rain mm"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower left", ncol=2)
        a.set_ylim(-27, -3)
    # 10 cloud-free chips across the year
    c = clr.copy(); c["m"] = c.date.dt.to_period("M")
    ds = list(c.groupby("m").apply(lambda q: q.date.iloc[len(q) // 2], include_groups=False))
    if len(ds) > 10:
        ds = [ds[i] for i in np.linspace(0, len(ds) - 1, 10).round().astype(int)]
    for k, (d, ch) in enumerate(zip(ds, chips(r.row, r.col, ds, ("B5", "B3", "B2"), 40))):
        a = fig.add_subplot(gs[4, k]); a.imshow(ch); a.add_patch(plt.Rectangle((18.5, 18.5), 2, 2, fill=False, ec="red", lw=1.2))
        a.set_title(f"{d:%Y-%m-%d}", fontsize=8); a.axis("off")
    # context chips: clearest dry-season date (or any), 128 px = 1.28 km
    dry = clr[(clr.date >= "2026-01-01") & (clr.date <= "2026-04-30")]
    dctx = (dry if len(dry) else clr).sort_values("cs_cdf").date.iloc[-1] if len(clr) else None
    if dctx is not None:
        for k, (bands, lab) in enumerate([(("B5", "B3", "B2"), "B5/B3/B2"), (("B8", "B4", "B3"), "B8/B4/B3 (veg = red)")]):
            a = fig.add_subplot(gs[5, 5 * k:5 * k + 5])
            a.imshow(chips(r.row, r.col, [dctx], bands, 128, "chip")[0])
            a.add_patch(plt.Rectangle((62.5, 62.5), 2, 2, fill=False, ec="yellow", lw=1.5))
            a.set_title(f"context 1.28 km, {lab}, {dctx:%Y-%m-%d}", fontsize=9); a.axis("off")
    fig.savefig(A / "sheets" / f"{r.audit_id}.png", dpi=66, bbox_inches="tight")
    plt.close(fig)


def _draw(chunk, T):
    dem = rasterio.open(PROJ / "data/ancillary/dem.tif")
    for r in chunk.itertuples():
        sheet(r, T, dem)


def main():
    (A / "sheets").mkdir(parents=True, exist_ok=True); (A / "key").mkdir(exist_ok=True)
    d = sample()
    d.to_csv(A / "key/audit_key.csv", index=False)
    d[["audit_id", "lon", "lat"]].to_csv(A / "points.csv", index=False)
    print(d.groupby("bin").size().to_dict(), flush=True)
    T = templates()
    dem = rasterio.open(PROJ / "data/ancillary/dem.tif")
    # draw sheets in parallel processes (each opens its own rasters)
    from concurrent.futures import ProcessPoolExecutor
    from safe_resources import workers
    n = workers(1.5 * 2**30)          # ~1.5 GB per sheet process; sized by the pod's memory REQUEST, not its limit
    print(f"{n} sheet workers", flush=True)
    chunks = [d.iloc[i::n] for i in range(n)]
    with ProcessPoolExecutor(n) as pool:
        list(pool.map(_draw, chunks, [T] * n))
    print("sheets:", len(list((A / "sheets").glob("*.png"))))


if __name__ == "__main__":
    main()

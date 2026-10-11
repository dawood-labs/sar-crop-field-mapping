"""Step 2 of the GCP EDA: figures from data/gcps/eda/gcp_series.pkl + gcp_features.csv.

data/gcps/eda/sheets/<gcp_id>_<crop>.png   one sheet per point: NDVI (raw / cloud-free / Whittaker), VH and VV of both
                                            tracks (3x3 window) with rain, 10 cloud-free B5/B3/B2 chips across the year
data/gcps/eda/montage_<crop>_<k>.png        12 points per page, same class: NDVI + VH mini panels and one dry-season chip
data/gcps/eda/class_summary.png             per class: median and IQR of Whittaker NDVI and of VH per track
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
from rasterio.windows import Window

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import explorer as ex  # noqa: E402

EDA = PROJ / "data/gcps/eda"
TC = ex.TRACK_COLORS
STRETCH = json.loads((PROJ / "data/stacks/s2_stretch.json").read_text())
S2_DATES = None


def chips_for(row, col, dates, size=40):
    global S2_DATES
    if S2_DATES is None:
        S2_DATES = list(pd.to_datetime(ex._dates(PROJ / "data/stacks/S2_NDVI.tif")))
    idx = [S2_DATES.index(d) + 1 for d in dates]
    h = size // 2
    win = Window(col - h, row - h, size, size)
    lay = [ex._ds(PROJ / f"data/stacks/S2_{b}.tif").read(idx, window=win, boundless=True, fill_value=0).astype("float32")
           for b in ("B5", "B3", "B2")]
    out = []
    for k in range(len(idx)):
        out.append(np.dstack([np.clip((L[k] - STRETCH[b][0]) / (STRETCH[b][1] - STRETCH[b][0]), 0, 1)
                              for b, L in zip(("B5", "B3", "B2"), lay)]))
    return out


def pick_dates(s2, n=10):
    """Up to n cloud-free dates spread over the year (one per ~month where possible)."""
    c = s2[s2.clear].copy()
    if c.empty:
        return []
    c["m"] = c.date.dt.to_period("M")
    per_month = c.groupby("m").apply(lambda g: g.iloc[len(g) // 2], include_groups=False)
    d = list(per_month.date)
    if len(d) > n:
        d = [d[i] for i in np.linspace(0, len(d) - 1, n).round().astype(int)]
    return d


def sheet(gid, s, f):
    s2, s1, rain = s["s2"], s["s1_3x3"], s["rain"]
    dd, z = ex.whittaker(s2.date, s2.ndvi.values, s2.clear.astype(float).values, 1.0, 2)
    dates = pick_dates(s2)
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(5, 10, height_ratios=[0.35, 1.3, 1.1, 1.1, 1.0], hspace=0.45)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    ax0.text(0, 0.95, f"{gid}   label: {s['crop']}   lon {s['lon']:.5f}  lat {s['lat']:.5f}   row {s['row']} col {s['col']}",
             fontsize=14, weight="bold", va="top")
    fv = str(f.get("final_verdict", f.get("verdict", "-")))
    note = f.get("visual_note") if isinstance(f.get("visual_note"), str) else f.get("reasons", "")
    ax0.text(0, 0.45, f"VERDICT: {fv}   (rules: {f.get('verdict', '-')})   class 2026: {f.get('class_2026', '-')}", fontsize=11, va="top",
             color="green" if fv.startswith("OK") else {"Doubtful": "darkorange", "Likely wrong": "red"}.get(fv, "black"))
    ax0.text(0, 0.0, str(note)[:260], fontsize=9, va="top", wrap=True)
    a1 = fig.add_subplot(gs[1, :])
    a1.plot(s2.date, s2.ndvi, color="lightgray", marker="x", ms=4, lw=1, label="NDVI raw (all dates)")
    c = s2[s2.clear]
    a1.plot(c.date, c.ndvi, "o-", color="green", ms=5, lw=1.2, label=f"cloud-free ({len(c)})")
    a1.plot(dd, z, color="royalblue", lw=2.2, label="Whittaker (lambda 1, d 2)")
    for d in dates:
        a1.axvline(d, color="orange", alpha=0.25, lw=3)
    a1.set_ylim(-0.1, 1); a1.set_ylabel("NDVI"); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="upper right", ncol=3)
    for i, var in enumerate(["VH", "VV"]):
        a = fig.add_subplot(gs[2 + i, :], sharex=a1); a2 = a.twinx()
        a2.bar(rain.index, rain.best, width=1, color="lightgray", alpha=0.6)
        for t, g in s1.groupby("track"):
            a2.bar(g.date, g.rain24h_imerg, width=2, color=TC.get(t), alpha=0.25)
            a.plot(g.date, g[var], "o-", ms=3, lw=1.2, color=TC.get(t), label=f"{t} {var} (3x3)")
        if var == "VH":
            a.axhline(-20, color="k", ls=":", lw=1); a.text(s2.date.min(), -20.3, " -20 dB (standing water)", fontsize=7, va="top")
        a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
        a.set_ylabel(f"{var} dB"); a2.set_ylabel("rain mm"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
    if dates:
        for k, (d, ch) in enumerate(zip(dates, chips_for(s["row"], s["col"], dates))):
            a = fig.add_subplot(gs[4, k]); a.imshow(ch); a.add_patch(plt.Rectangle((18.5, 18.5), 2, 2, fill=False, ec="red", lw=1.3))
            a.set_title(f"{d:%Y-%m-%d}", fontsize=8); a.axis("off")
    fig.savefig(EDA / "sheets" / f"{gid}_{s['crop'].replace(' ', '_')}.png", dpi=72, bbox_inches="tight")
    plt.close(fig)


def montage(crop, ids, series, feats, page):
    n = len(ids); nc = 4; nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc * 2, figsize=(26, 3.1 * nr), gridspec_kw={"width_ratios": [3, 1] * nc}, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, gid in enumerate(ids):
        s, f = series[gid], feats.loc[gid]
        a = axes[k // nc, 2 * (k % nc)]; a.axis("on")
        s2 = s["s2"]; c = s2[s2.clear]
        dd, z = ex.whittaker(s2.date, s2.ndvi.values, s2.clear.astype(float).values, 1.0, 2)
        a.plot(c.date, c.ndvi, "o", color="green", ms=3); a.plot(dd, z, color="royalblue", lw=1.8)
        a.set_ylim(-0.1, 1); a.tick_params(labelsize=7); a.grid(alpha=0.3)
        b = a.twinx()
        for t, g in s["s1_3x3"].groupby("track"):
            b.plot(g.date, g.VH, lw=0.9, color=TC.get(t), alpha=0.8)
        b.axhline(-20, color="k", ls=":", lw=0.8); b.set_ylim(-27, -8); b.tick_params(labelsize=7)
        fv = str(f.get("final_verdict", f.get("verdict", "")))
        col = "green" if fv.startswith("OK") else {"Doubtful": "darkorange", "Likely wrong": "red"}.get(fv, "black")
        a.set_title(f"{gid} {s['crop']} | {fv}", fontsize=9, color=col)
        dry = s2[s2.clear & (s2.date >= "2026-01-15") & (s2.date <= "2026-03-31")]
        if len(dry):
            d = dry.date.iloc[len(dry) // 2]
            ch = chips_for(s["row"], s["col"], [d])[0]
            ac = axes[k // nc, 2 * (k % nc) + 1]; ac.imshow(ch); ac.axis("off")
            ac.add_patch(plt.Rectangle((18.5, 18.5), 2, 2, fill=False, ec="red", lw=1.2)); ac.set_title(f"{d:%Y-%m-%d}", fontsize=7)
    fig.suptitle(f"{crop}: green = cloud-free NDVI, blue = Whittaker (left axis); lines = VH 3x3, "
                 f"blue RO164 DES / red RO099 ASC (right axis, dotted -20 dB); chip = dry season B5/B3/B2", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(EDA / f"montage_{crop.replace(' ', '_')}_{page}.png", dpi=62)
    plt.close(fig)


def class_summary(series, feats):
    crops = feats.crop.value_counts().index
    fig, axes = plt.subplots(len(crops), 3, figsize=(20, 2.4 * len(crops)), sharex=True, squeeze=False)
    day = pd.date_range("2025-09-01", "2026-10-08")
    for i, crop in enumerate(crops):
        ids = feats.index[feats.crop == crop]
        W = []
        for gid in ids:
            s2 = series[gid]["s2"]
            dd, z = ex.whittaker(s2.date, s2.ndvi.values, s2.clear.astype(float).values, 1.0, 2)
            W.append(pd.Series(z, index=dd).reindex(day).values)
        W = np.array(W)
        a = axes[i, 0]
        for w in W:
            a.plot(day, w, color="royalblue", alpha=0.15, lw=0.8)
        a.plot(day, np.nanmedian(W, 0), color="navy", lw=2)
        a.set_ylim(-0.1, 1); a.set_ylabel(f"{crop}\n(n={len(ids)})", fontsize=9); a.grid(alpha=0.3)
        for j, t in enumerate(["RO164_DES", "RO099_ASC"]):
            a = axes[i, 1 + j]
            S = []
            for gid in ids:
                g = series[gid]["s1_3x3"]; g = g[g.track == t].set_index("date").VH
                S.append(g)
            S = pd.concat(S, axis=1)
            q = S.quantile([0.25, 0.5, 0.75], axis=1).T
            a.fill_between(q.index, q[0.25], q[0.75], color=TC[t], alpha=0.2); a.plot(q.index, q[0.5], color=TC[t], lw=1.8)
            a.axhline(-20, color="k", ls=":", lw=0.8); a.set_ylim(-26, -9); a.grid(alpha=0.3)
            if i == 0:
                a.set_title(f"VH {t} (3x3), median + IQR")
    axes[0, 0].set_title("NDVI Whittaker (each point + median)")
    fig.tight_layout(); fig.savefig(EDA / "class_summary.png", dpi=70); plt.close(fig)


def main(which="all"):
    series = pickle.load(open(EDA / "gcp_series.pkl", "rb"))
    vf = EDA / "gcp_verdicts.csv"
    feats = pd.read_csv(vf if vf.exists() else EDA / "gcp_features.csv").set_index("gcp_id")
    (EDA / "sheets").mkdir(exist_ok=True)
    if which in ("all", "summary"):
        class_summary(series, feats)
    if which in ("all", "montage"):
        for p in EDA.glob("montage_*.png"):
            p.unlink()
        for crop, g in feats.groupby("crop"):
            ids = list(g.index)
            for k in range(0, len(ids), 12):
                montage(crop, ids[k:k + 12], series, feats, k // 12 + 1)
    if which in ("all", "montage"):
        flagged = list(feats.index[~feats.verdict.isin(["OK", "OK (no clear flooding)", "OK (weak test)", "No label"])])  # rule-flagged
        for k in range(0, len(flagged), 12):
            montage("FLAGGED", flagged[k:k + 12], series, feats, k // 12 + 1)
    if which in ("all", "sheets"):
        for gid in feats.index:
            sheet(gid, series[gid], feats.loc[gid].to_dict())
    print("figures written to", EDA)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")

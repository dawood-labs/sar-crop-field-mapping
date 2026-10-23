"""Blind review of an external test site: are the model's extra "cane" fields (label non-cane, map cane) real cane?

  python scripts/site_review.py sites/<name> sample   -> <site>/review/: sheets/R###.png, RUBRIC.md, key/key.csv,
                                                         key/samples.parquet (never shown to reviewers)
  python scripts/site_review.py sites/<name> compare  -> review_results.csv + summary.json
Groups (one px per field-like patch: >= 1 acre, >= 2 px from the patch edge, patch centre):
  map_cane_label_non 40, label_cane_map_non 20, both_cane 10, both_non 10.
Sheets as in qc_review.py: NIR false-colour chips (B8/B4/B3) on cloud-free dates (Cloud Score+ cs_cdf >= 0.60 on the
3 x 3 window), NDVI on those dates (Nov 2024 - Nov 2025), 15-day S1 VH and VH-VV (Jan - Oct 2025).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

import qc_review as QR  # noqa: E402  (rubric text + stretch)
import site_paths as SP

YEAR = int(os.environ.get("REVIEW_YEAR", "2025"))
if os.environ.get("REVIEW_MODE") == "change":     # 2025 -> 2026 change raster (same coding as map_vs_label: 0..3)
    GROUPS = [("new_2026_only", 3, 20), ("lost_2025_only", 2, 40), ("cane_both", 1, 10), ("noncane_both", 0, 10)]
else:
    GROUPS = [("map_cane_label_non", 3, 30), ("label_cane_map_non", 2, 30), ("both_cane", 1, 10), ("both_non", 0, 10)]
S2_START, S2_END = pd.Timestamp(f"{YEAR - 1}-11-01"), pd.Timestamp(f"{YEAR}-11-30")
FILE_YEAR = os.environ.get("FILE_YEAR", "2025")   # year in the map file names (apply_site.py YEAR=2026 -> 2026)
RUBRIC_FILE = os.environ.get("RUBRIC_FILE")      # e.g. the 2026 rubric
CHIP = 64


TAG = os.environ.get("OUT_TAG", "")      # model sub-folder, e.g. multi2025 (review goes to review_<tag>/)


def sample(site):
    from scipy import ndimage
    out = site / ("review_" + TAG if TAG else "review")
    for d in ("sheets", "key", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    with rasterio.open(site / "data/model" / TAG / f"map_vs_label_{FILE_YEAR}.tif") as r:
        c, tr, crs = r.read(1), r.transform, r.crs
    p = rasterio.open(site / "data/model" / TAG / f"cane_prob_{FILE_YEAR}.tif").read(1)
    rng = np.random.default_rng(5)
    rows = []
    for g, v, n in GROUPS:
        m = c == v
        L, k = ndimage.label(m, structure=np.ones((3, 3)))
        sz = np.bincount(L.ravel()); sz[0] = 0
        D = ndimage.distance_transform_edt(m)
        dmax = np.asarray(ndimage.maximum(D, L, np.arange(k + 1)))
        ok = np.flatnonzero((sz >= 40) & (dmax >= 2))
        pick = rng.choice(ok, min(n, len(ok)), replace=False)
        for q, (r_, c_) in zip(pick, ndimage.maximum_position(D, L, pick)):
            rows.append({"group": g, "row": int(r_), "col": int(c_), "patch_px": int(sz[q]), "p": int(p[r_, c_])})
        print(f"  {g}: {len(ok)} field-like patches, {len(pick)} sampled", flush=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=5).reset_index(drop=True)
    df["sample_id"] = [f"R{i + 1:03d}" for i in range(len(df))]
    df["pixel_id"] = df.row * c.shape[1] + df.col
    import geopandas as gpd
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(tr.c + (df.col + .5) * tr.a, tr.f + (df.row + .5) * tr.e), crs=crs)
    ll = g.to_crs(4326).geometry; df["lon"], df["lat"] = ll.x.round(6), ll.y.round(6)
    df.to_csv(out / "key/key.csv", index=False); g.to_parquet(out / "key/samples.parquet")
    (out / "RUBRIC.md").write_text(Path(RUBRIC_FILE).read_text() if RUBRIC_FILE else QR.RUBRIC)
    import multiprocessing as mp
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(sheet, [(str(site), j, out.name) for j in df[["sample_id", "row", "col"]].to_dict("records")], chunksize=4))
    print(f"{len(df)} sheets -> {out / 'sheets'}")


def sheet(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    site, job = SP.site_dir(args[0]), args[1]
    ST = site / "data/stacks"
    row, col, sid, h = job["row"], job["col"], job["sample_id"], CHIP // 2
    w3 = Window(col - 1, row - 1, 3, 3)
    with rasterio.open(ST / "S2_cs_cdf.tif") as r:
        dates = pd.to_datetime(list(r.descriptions))
        cs = r.read(window=w3, boundless=True, fill_value=65535).reshape(len(dates), -1)
    clear = ((cs >= 60) & (cs <= 100)).mean(1) >= 0.67
    b8 = rasterio.open(ST / "S2_B8.tif").read(window=w3, boundless=True, fill_value=0).reshape(len(dates), -1).astype("float32")
    b4 = rasterio.open(ST / "S2_B4.tif").read(window=w3, boundless=True, fill_value=0).reshape(len(dates), -1).astype("float32")
    ndvi = ((b8 - b4) / np.maximum(b8 + b4, 1)).mean(1)
    ok = clear & (dates >= S2_START) & (dates <= S2_END) & (b8.mean(1) > 0)
    idx = list(np.flatnonzero(ok))
    if len(idx) > 16:
        idx = [idx[int(j)] for j in np.linspace(0, len(idx) - 1, 16).round()]
    win = Window(col - h, row - h, CHIP, CHIP)
    L = {b: rasterio.open(ST / f"S2_{b}.tif").read([i + 1 for i in idx], window=win, boundless=True, fill_value=0).astype("float32")
         for b in ("B8", "B4", "B3")} if idx else {}
    chips = [np.dstack([np.clip((L[b][k] - QR.STRETCH[b][0]) / (QR.STRETCH[b][1] - QR.STRETCH[b][0]), 0, 1) for b in ("B8", "B4", "B3")])
             for k in range(len(idx))]
    with rasterio.open(site / f"data/features/s1_15d_{YEAR}jan.tif") as r:
        names = list(r.descriptions)
        v = r.read(window=Window(col, row, 1, 1)).ravel().astype("float32") / 100
    s1 = pd.DataFrame([{"track": "_".join(n.split("_")[:2]), "var": n.split("_")[2], "p": n.split("_")[3], "dB": x} for n, x in zip(names, v)])
    s1["date"] = [pd.Timestamp(f"{q[:4]}-{q[4:6]}-{'01' if q[6] == 'a' else '16'}") + pd.Timedelta(days=7) for q in s1.p]
    ncols, nr = 8, max(1, int(np.ceil(len(chips) / 8)))
    fig = plt.figure(figsize=(16, 2.2 * nr + 7.5))
    gs = fig.add_gridspec(nr + 3, ncols, height_ratios=[2.2] * nr + [2.3, 2.6, 2.6])
    for k, ch in enumerate(chips):
        ax = fig.add_subplot(gs[k // ncols, k % ncols]); ax.imshow(ch, interpolation="nearest")
        ax.add_patch(plt.Rectangle((h - 1.5, h - 1.5), 3, 3, fill=False, ec="red", lw=1.5))
        ax.set_title(f"{dates[idx[k]]:%Y-%m-%d}", fontsize=9); ax.set_axis_off()
    xl = (S2_START, S2_END)
    ax = fig.add_subplot(gs[nr, :]); ax.plot(dates[ok], ndvi[ok], "o-", color="green", ms=4)
    ax.set_ylim(-0.1, 1); ax.set_xlim(xl); ax.set_ylabel("NDVI (clear)"); ax.grid(alpha=.3)
    for i, var in enumerate(("VH", "VHmVV")):
        ax = fig.add_subplot(gs[nr + 1 + i, :])
        for t, g in s1[s1["var"] == var].groupby("track"):
            ax.plot(g.date, g.dB, "o-", ms=3, label=t, color={"RO164_DES": "#1f77b4", "RO099_ASC": "#d62728"}.get(t))
        ax.set_xlim(xl); ax.set_ylabel(f"S1 {'VH' if var == 'VH' else 'VH-VV'} (dB)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(f"{sid}   (red box = the pixel, 3x3 px; chips 640 m wide, NIR false colour B8/B4/B3; radar Jan-Oct {YEAR} only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(site / args[2] / "sheets" / f"{sid}.png", dpi=62); plt.close(fig)


def compare(site):
    out = site / ("review_" + TAG if TAG else "review")
    key = pd.read_csv(out / "key/key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((out / "labels").glob("batch_*.csv"))], ignore_index=True)
    d = key.merge(lab, on="sample_id", how="left")
    d.to_csv(out / "review_results.csv", index=False)
    s = {g: {"n": int(len(x)), **x.answer.value_counts().to_dict(), "what": x.what.value_counts().to_dict()} for g, x in d.groupby("group")}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    site = SP.site_dir(sys.argv[1])
    {"sample": sample, "compare": compare}[sys.argv[2]](site)

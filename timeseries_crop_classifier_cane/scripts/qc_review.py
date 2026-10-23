"""Blind review of the label-QC flags: are flagged pixels really label errors?

  python scripts/qc_review.py sample    -> data/labels/qc_review/: sheets/R###.png (what reviewers see), RUBRIC.md,
                                           key/key.csv + key/samples.parquet (group, label, p_oof: NEVER shown to
                                           reviewers), reports/label_qc/ref_curves.csv (median clean curves for the
                                           pixel explorer)
  python scripts/qc_review.py compare   -> reads labels/batch_*.csv (sample_id, answer, what, confidence, note),
                                           writes review_results.csv + summary.json: per flag group, how often the
                                           reviewers say the 2025 label is wrong
Groups (one pixel per 500 m block, interior pixels only, edge >= 2): cl_cane_low 40, cl_noncane_high 40,
outlier_unit 35, s2_not_green 35, clean_cane 25, clean_noncane 25 (the clean groups calibrate the reviewers).
A sheet shows NIR false-colour S2 chips (B8/B4/B3, 64 x 64 px, red box = 3 x 3 px around the pixel) on cloud-free
dates Sep 2024 - Nov 2025, the cloud-free NDVI of the pixel and its 15-day S1 VH / VH-VV curves. No label, no flag,
no model output.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os, time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
ST = PROJ / "data/stacks"
ROUND = os.environ.get("QC_ROUND", "1")       # round 2 = double check after the outlier fix
OUT = PROJ / ("data/labels/qc_review" if ROUND == "1" else f"data/labels/qc_review_r{ROUND}")
QC = PROJ / "data/labels/cane2025_qc.tif"
FEAT = PROJ / "data/features/s1_15d_2025.tif"
SEED, CHIP = (1 if ROUND == "1" else 7), 64
GROUPS = ([("cl_cane_low", 40), ("cl_noncane_high", 40), ("outlier_unit", 35), ("s2_not_green", 35),
           ("clean_cane", 25), ("clean_noncane", 25)] if ROUND == "1" else
          [("cl_cane_low", 30), ("cl_noncane_high", 30), ("outlier_unit", 25), ("s2_not_green", 15),
           ("clean_cane", 10), ("clean_noncane", 10)] if ROUND == "2" else
          [("cl_cane_low", 40), ("cl_noncane_high", 40), ("s2_not_green", 10), ("clean_cane", 10), ("clean_noncane", 10)])
# round 3 (user): only field-like flag patches (>= 1 acre, core >= 2 px from the patch edge), the pixel = patch centre
PATCH_MIN_PX, PATCH_CORE = 40, 2
S2_START, S2_END = pd.Timestamp("2024-09-01"), pd.Timestamp("2025-11-30")
STRETCH = {"B8": (1000, 4500), "B4": (200, 2200), "B3": (300, 1800)}

RUBRIC = """# Blind review: was this pixel SUGARCANE in the 2025 season?

Each sheet = one pixel (the RED BOX, 3 x 3 pixels = 30 m, in the centre of every chip). Decide what the land in the
box was in the 2025 season, i.e. what stood there at the END OF NOVEMBER 2025.

What the sheet shows
- Top: Sentinel-2 NIR false colour chips (B8 / B4 / B3), 640 m wide, on the dates the pixel was cloud-free.
  Bright red = dense green vegetation; pink / dull red = sparse or young vegetation; cyan / grey / brown = bare soil,
  stubble, built-up; dark blue / black = water.
- Middle: NDVI of the pixel on cloud-free dates (Sep 2024 - Nov 2025).
- Bottom: Sentinel-1 radar, 15-day composites: VH (dB) and VH-VV (dB), two orbits. Higher VH = more vegetation
  volume; a deep VH drop = harvest / bare field.

Typical patterns in this area (NE the country, rainfed uplands and lowlands)
- SUGARCANE: harvested Dec - Apr (field turns bare / cyan, NDVI ~0.2-0.35, VH drops to about -21 / -22 dB), then
  the ratoon REGROWS QUICKLY even in the dry season (NDVI rising Mar - May), tall dense canopy Jun - Nov
  (NDVI 0.7 - 0.85, bright red, VH high and stable). Fields are medium/large blocks with straight edges. A field
  planted Oct - Dec 2024 shows bare soil then, then grows through 2025 without a harvest dip in early 2025.
  PLANT CANE planted Oct - Dec 2024 is common here: bare Oct - Nov 2024, then steady growth through the dry season
  (NDVI 0.4 in Jan -> 0.7-0.8 by Mar - Apr) and dense until Nov 2025, with NO dip in early 2025. That is cane.
- TWO CROPS A YEAR (irrigated dry-season crop + wet-season rice, often next to canals): a green peak Feb - Apr, a DROP
  to NDVI ~0.3 in late Apr - May, then a second crop Jun - Nov, often with dark flooded plots Jun - Sep. Not cane:
  cane never drops in May once it has regrown.
- CASSAVA: planted mostly Mar - May or Oct - Nov, canopy less dense than cane (NDVI peak ~0.6 - 0.75, duller red),
  harvested at scattered times; often small fields next to cane. Hard to separate from cane: say so in the note.
- RICE (lowland paddies, small irregular plots with bunds): green Jul - Oct, harvested Oct - Dec, then bare/dry
  with NO regrowth Jan - Apr (NDVI stays low), flooded dark plots in Jun - Aug.
- TREES / ORCHARD / EUCALYPTUS / FOREST: high NDVI (>= 0.6) all year, also Feb - Apr; no harvest dip; rows may show.
- GRASS / FALLOW / PASTURE: low-to-medium NDVI, no clear crop cycle.
- WATER, BUILT-UP, ROADS, BARE: obvious from the chips.
- MIXED / EDGE: the box sits on a boundary between two land covers.

Answer one row per sheet in your CSV: sample_id,answer,what,confidence,note
- answer: cane | not_cane | unsure
- what: sugarcane | cassava | rice | other_crop | tree | grass_fallow | water | built | bare | mixed_edge | unknown
- confidence: high | medium | low
- note: a few words (what you saw and when)
Judge only from the sheet. Do not guess from other sheets; every pixel is independent.
"""


def sheets_only():
    t0 = time.time()
    df = pd.read_csv(OUT / "key/key.csv")
    import multiprocessing as mp
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(sheet, df[["sample_id", "row", "col"]].to_dict("records"), chunksize=4))
    print(f"{len(df)} sheets ({time.time() - t0:.0f} s)", flush=True)


def sample():
    t0 = time.time()
    (OUT / "sheets").mkdir(parents=True, exist_ok=True); (OUT / "key").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    with rasterio.open(QC) as r:
        clean, flags, block, p_oof, edge = r.read(1), r.read(2), r.read(3), r.read(5), r.read(7)
        tr, crs = r.transform, r.crs
    lab = rasterio.open(PROJ / "data/labels/cane2025_labels.tif").read(1)
    inner = edge >= 2
    masks = {"cl_cane_low": (flags & 128) > 0, "cl_noncane_high": (flags & 256) > 0,
             "outlier_unit": ((flags & 1) > 0) & ((flags & 128) == 0),
             "s2_not_green": ((flags & 32) > 0) & ((flags & 128) == 0) & ((flags & 1) == 0),
             "clean_cane": clean == 1, "clean_noncane": clean == 0}
    rng = np.random.default_rng(SEED)
    used_blocks, rows = set(), []
    if ROUND != "1":
        used_blocks |= set(pd.read_csv(PROJ / "data/labels/qc_review/key/key.csv").block_id)
    for g, n in GROUPS:
        if ROUND == "3":
            rows += patch_pixels(g, masks[g], n, rng, block, lab, p_oof, flags)
            continue
        rr, cc = np.nonzero(masks[g] & inner)
        order = rng.permutation(len(rr))
        k = 0
        for i in order:
            b = int(block[rr[i], cc[i]])
            if b in used_blocks:
                continue
            used_blocks.add(b)
            rows.append({"group": g, "row": int(rr[i]), "col": int(cc[i]), "block_id": b,
                         "label_2025": int(lab[rr[i], cc[i]]), "p_oof": int(p_oof[rr[i], cc[i]]), "flags": int(flags[rr[i], cc[i]])})
            k += 1
            if k == n:
                break
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["sample_id"] = [f"R{i + 1:03d}" for i in range(len(df))]
    W = lab.shape[1]
    df["pixel_id"] = df.row * W + df.col
    df["x"] = tr.c + (df.col + 0.5) * tr.a; df["y"] = tr.f + (df.row + 0.5) * tr.e
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=crs)
    ll = gdf.to_crs(4326); df["lon"], df["lat"] = ll.geometry.x.round(6), ll.geometry.y.round(6)
    df.drop(columns="x").drop(columns="y").to_csv(OUT / "key/key.csv", index=False)
    gdf.to_parquet(OUT / "key/samples.parquet")
    (OUT / "RUBRIC.md").write_text(RUBRIC)
    ref_curves(clean, edge, rng)
    print(df.group.value_counts().to_string(), f"\nsampled ({time.time() - t0:.0f} s)", flush=True)
    jobs = df[["sample_id", "row", "col"]].to_dict("records")
    import multiprocessing as mp   # spawn: forked workers hang on GDAL's thread pool
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(sheet, jobs, chunksize=4))
    print(f"{len(jobs)} sheets -> {OUT / 'sheets'} ({time.time() - t0:.0f} s)", flush=True)


def patch_pixels(g, m, n, rng, block, lab, p_oof, flags):
    """One pixel (the most interior one) per field-like patch of the mask, n patches at random."""
    from scipy import ndimage
    L, k = ndimage.label(m, structure=np.ones((3, 3)))
    sz = np.bincount(L.ravel()); sz[0] = 0
    D = ndimage.distance_transform_edt(m)
    idx = np.arange(k + 1)
    dmax = np.asarray(ndimage.maximum(D, L, idx))
    ok = np.flatnonzero((sz >= PATCH_MIN_PX) & (dmax >= PATCH_CORE))
    pick = rng.choice(ok, min(n, len(ok)), replace=False)
    pos = ndimage.maximum_position(D, L, pick)
    print(f"  {g}: {len(ok)} field-like patches, {len(pick)} sampled", flush=True)
    return [{"group": g, "row": int(r), "col": int(c), "block_id": int(block[r, c]), "label_2025": int(lab[r, c]),
             "p_oof": int(p_oof[r, c]), "flags": int(flags[r, c]), "patch_px": int(sz[q])} for q, (r, c) in zip(pick, pos)]


def ref_curves(clean, edge, rng, n=20000):
    out = []
    with rasterio.open(FEAT) as r:
        names = list(r.descriptions)
        for cls, name in ((1, "cane"), (0, "noncane")):
            rr, cc = np.nonzero((clean == cls) & (edge >= 2))
            k = rng.choice(len(rr), min(n, len(rr)), replace=False)
            X = np.stack([r.read(i + 1)[rr[k], cc[k]] for i in range(len(names))], 1).astype("float32") / 100
            med = np.median(X, 0)
            for nm, v in zip(names, med):
                t = nm.split("_"); p = t[3]
                d = pd.Timestamp(f"{p[:4]}-{p[4:6]}-{'01' if p[6] == 'a' else '16'}") + pd.Timedelta(days=7)
                out.append({"cls": name, "track": f"{t[0]}_{t[1]}", "var": t[2], "period": p, "date": d.date(), "dB": round(float(v), 2)})
    (PROJ / "reports/label_qc").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(PROJ / "reports/label_qc/ref_curves.csv", index=False)


def sheet(job):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    row, col, sid = job["row"], job["col"], job["sample_id"]
    h = CHIP // 2
    with rasterio.open(ST / "S2_CLEAR.tif") as r:
        dates = pd.to_datetime(list(r.descriptions))
        clr = r.read(window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=255).reshape(len(dates), -1)
    clear = (clr == 1).mean(1) >= 0.67
    with rasterio.open(ST / "S2_NDVI.tif") as r:
        nd = r.read(window=Window(col - 1, row - 1, 3, 3), boundless=True, fill_value=-32768).reshape(len(dates), -1).astype("float32")
    nd[nd == -32768] = np.nan
    ndvi = np.nanmean(nd, 1) / 1e4
    ok = clear & (dates >= S2_START) & (dates <= S2_END)
    idx = np.flatnonzero(ok)
    # at most 2 chips per month, 16 in total, spread over the season
    keep, per_m = [], {}
    for i in idx:
        m = dates[i].strftime("%Y%m")
        if per_m.get(m, 0) < 2:
            keep.append(i); per_m[m] = per_m.get(m, 0) + 1
    if len(keep) > 16:
        keep = [keep[int(j)] for j in np.linspace(0, len(keep) - 1, 16).round()]
    chips = []
    if keep:
        win = Window(col - h, row - h, CHIP, CHIP)
        L = {b: rasterio.open(ST / f"S2_{b}.tif").read([i + 1 for i in keep], window=win, boundless=True, fill_value=0).astype("float32")
             for b in ("B8", "B4", "B3")}
        for k in range(len(keep)):
            chips.append(np.dstack([np.clip((L[b][k] - STRETCH[b][0]) / (STRETCH[b][1] - STRETCH[b][0]), 0, 1) for b in ("B8", "B4", "B3")]))
    with rasterio.open(FEAT) as r:
        names = list(r.descriptions)
        v = r.read(window=Window(col, row, 1, 1)).ravel().astype("float32") / 100
    s1 = pd.DataFrame([{"track": f"{n.split('_')[0]}_{n.split('_')[1]}", "var": n.split("_")[2], "p": n.split("_")[3], "dB": x}
                       for n, x in zip(names, v)])
    s1["date"] = [pd.Timestamp(f"{p[:4]}-{p[4:6]}-{'01' if p[6] == 'a' else '16'}") + pd.Timedelta(days=7) for p in s1.p]
    ncols = 8
    nr = max(1, int(np.ceil(len(chips) / ncols)))
    fig = plt.figure(figsize=(16, 2.2 * nr + 7.5))
    gs = fig.add_gridspec(nr + 3, ncols, height_ratios=[2.2] * nr + [2.3, 2.6, 2.6])
    for k, c in enumerate(chips):
        ax = fig.add_subplot(gs[k // ncols, k % ncols]); ax.imshow(c, interpolation="nearest")
        ax.add_patch(plt.Rectangle((h - 1.5, h - 1.5), 3, 3, fill=False, ec="red", lw=1.5))
        ax.set_title(f"{dates[keep[k]]:%Y-%m-%d}", fontsize=9); ax.set_axis_off()
    if not chips:
        ax = fig.add_subplot(gs[0, :]); ax.text(0.5, 0.5, "no cloud-free S2 date", ha="center"); ax.set_axis_off()
    xl = (S2_START, S2_END)
    ax = fig.add_subplot(gs[nr, :])
    sel = clear & (dates >= S2_START) & (dates <= S2_END)
    ax.plot(dates[sel], ndvi[sel], "o-", color="green", ms=4); ax.set_ylim(-0.1, 1); ax.set_xlim(xl)
    ax.set_ylabel("NDVI (clear)"); ax.grid(alpha=0.3)
    for i, var in enumerate(("VH", "VHmVV")):
        ax = fig.add_subplot(gs[nr + 1 + i, :])
        for t, g in s1[s1["var"] == var].groupby("track"):
            ax.plot(g.date, g.dB, "o-", ms=3, label=t, color={"RO164_DES": "#1f77b4", "RO099_ASC": "#d62728"}.get(t))
        ax.set_xlim(xl); ax.set_ylabel(f"S1 {'VH' if var == 'VH' else 'VH-VV'} (dB)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"{sid}   (red box = the pixel, 3x3 px; chips 640 m wide, NIR false colour B8/B4/B3)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / "sheets" / f"{sid}.png", dpi=62); plt.close(fig)


def compare():
    key = pd.read_csv(OUT / "key/key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((OUT / "labels").glob("batch_*.csv"))], ignore_index=True)
    lab["sample_id"] = lab.sample_id.str.strip()
    d = key.merge(lab, on="sample_id", how="left")
    d["rev_cane"] = d.answer.map({"cane": 1, "not_cane": 0})
    d["label_wrong"] = np.where(d.rev_cane.isna(), np.nan, (d.rev_cane != d.label_2025).astype(float))
    d.to_csv(OUT / "review_results.csv", index=False)
    s = {}
    for g, x in d.groupby("group"):
        c = x[x.rev_cane.notna()]
        hc = c[c.confidence.isin(["high", "medium"])]
        s[g] = {"n": int(len(x)), "unsure_or_missing": int(x.rev_cane.isna().sum()),
                "label_wrong_pct": round(100 * c.label_wrong.mean(), 1) if len(c) else None,
                "label_wrong_pct_high_med": round(100 * hc.label_wrong.mean(), 1) if len(hc) else None,
                "what": x.what.value_counts().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    {"sample": sample, "sheets": sheets_only, "compare": compare}[sys.argv[1]]()

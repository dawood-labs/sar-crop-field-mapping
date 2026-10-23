"""Independent blind review of a model's 2026 maps in every AOI (user, 14 Oct 2026: "ek extensive review").

  python scripts/indep_review.py sample  [tag]  -> sites/indep_review_<tag>/: sheets/I###.png, RUBRIC.md, key/key.csv,
                                                    key/samples.parquet, google/ (basemap chips), never shown: key/
  python scripts/indep_review.py compare [tag]  -> review_results.csv, scores.csv (area-weighted P / R / F1 per AOI + ALL)
Sampling (no edge pixels): a 3 x 3 block is eligible only if the model map is one class over the 5 x 5 window around it
(block >= 1 px from any map boundary), the 2025 label is one class over the 3 x 3 block, and the block is uniform in
S2 (std of Dec and Feb NDVI over the 9 px <= 0.06). Per AOI N_CANE blocks mapped cane and N_NON mapped non-cane, at random.
Sheets show 3 x 3 MEAN and MEDIAN curves (S2 NDVI, S1 VH and VH-VV), S2 chips in 5-3-2 (B5/B3/B2: cane stands out) and
8-4-3 (B8/B4/B3), and a Google satellite chip (date unknown: context only). Reviewers never see the map or the label.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import io, json, math, os, urllib.request  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402
import site_paths as SP

PROJ = Path(__file__).resolve().parents[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "s2on"
OUT = SP.site_dir(os.environ.get)("OUT_NAME", f"indep_review_{TAG}")
# STRATA=change (fixed test set): strata = 2025 label x map of TAG, so changed areas (new / lost cane) get enough blocks;
# design weights stay valid for scoring ANY later map (read the map at the block, weight = stratum px / sampled)
STRATA = os.environ.get("STRATA", "map")
CHANGE_N = {"stable_cane": 10, "stable_non": 16, "lost": 7, "new": 7}     # (label, map) = (1,1) (0,0) (1,0) (0,1)
AOIS = ["site37_2026", "site32_2026", "site44_2026", "site11_2026", "site42_2026",
        "site24_2026", "site23_2026", "site05_2026", "site38_2026", "site25_2026",
        "bench_site27_2026", "bench_site15_2026"]
N_CANE, N_NON, CHIP = 12, 24, 64
SEED = int(os.environ.get("SEED", "14"))
EXCLUDE_FROM = [Path(x) for x in os.environ.get("EXCLUDE_FROM", "").split(",") if x]   # earlier reviews' key.csv: keep 300 m away
S2_START, S2_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31")
STRETCH = {"B8": (1000, 4500), "B4": (200, 2200), "B3": (300, 1800), "B5": (500, 2800), "B2": (150, 1500)}
ZOOM = 17


def eligible(site):
    from scipy import ndimage
    d = SP.site_dir(site) / "data"
    m = rasterio.open(d / f"model/{TAG}/cane_map_2026.tif").read(1)
    lab = rasterio.open(d / "labels/cane2025_labels.tif").read(1)
    k5, k3 = np.ones((5, 5), bool), np.ones((3, 3), bool)
    mv = np.where(m == 255, 0, m).astype("int16"); mv[m == 255] = 9          # outside / no data breaks uniformity
    map_ok = (ndimage.minimum_filter(mv, footprint=k5) == ndimage.maximum_filter(mv, footprint=k5)) & (m != 255)
    lv = lab.astype("int16")
    lab_ok = (ndimage.minimum_filter(lv, footprint=k3) == ndimage.maximum_filter(lv, footprint=k3)) & (lab != 255)
    with rasterio.open(d / "features/s2_dry_2026.tif") as r:
        names = list(r.descriptions)
        s2_ok = np.ones(m.shape, bool)
        for b in ("S2_NDVI_202512", "S2_NDVI_202602"):
            a = r.read(names.index(b) + 1).astype("float32"); a[a <= -32000] = np.nan; a /= 1000
            mu = ndimage.uniform_filter(np.nan_to_num(a), 3); mu2 = ndimage.uniform_filter(np.nan_to_num(a) ** 2, 3)
            sd = np.sqrt(np.maximum(mu2 - mu ** 2, 0))
            nanblk = ndimage.maximum_filter(np.isnan(a).astype("uint8"), footprint=k3) > 0
            s2_ok &= nanblk | (sd <= 0.06)
    ok = map_ok & lab_ok & s2_ok
    return m, lab, ok


def sample():
    for x in ("sheets", "key", "labels", "google"):
        (OUT / x).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for s in AOIS:
        m, lab, ok = eligible(s)
        for f in EXCLUDE_FROM:                                     # blocks already reviewed (maybe used in training)
            k = pd.read_csv(f); k = k[k.site == s]
            for r_, c_ in zip(k.row, k.col):
                ok[max(r_ - 30, 0):r_ + 31, max(c_ - 30, 0):c_ + 31] = False
        if STRATA == "change":
            groups = [(name, (lab == a) & (m == b), CHANGE_N[name]) for name, a, b in
                      (("stable_cane", 1, 1), ("stable_non", 0, 0), ("lost", 1, 0), ("new", 0, 1))]
        else:
            groups = [(name, m == cls, n) for cls, n, name in ((1, N_CANE, "map_cane"), (0, N_NON, "map_non"))]
        for name, sm, n in groups:
            rr, cc = np.nonzero(ok & sm)
            k = rng.choice(len(rr), min(n, len(rr)), replace=False)
            for r, c in zip(rr[k], cc[k]):
                rows.append({"site": s, "stratum": name, "row": int(r), "col": int(c), "pixel_id": int(r * m.shape[1] + c),
                             "label2025": int(lab[r, c]), "map_tag": int(m[r, c]), "n_stratum_px": int(sm.sum()),
                             "n_eligible_px": len(rr), "weight": sm.sum() / max(len(k), 1)})
        print(f"{s}: eligible cane {int((ok & (m == 1)).sum()):,} / non {int((ok & (m == 0)).sum()):,} px", flush=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["sample_id"] = [f"{os.environ.get('ID_PREFIX', 'I')}{i + 1:03d}" for i in range(len(df))]
    import geopandas as gpd
    tr = {s: rasterio.open(SP.site_dir(s) / "data/model" / TAG / "cane_map_2026.tif").transform for s in AOIS}
    x = [tr[r.site].c + (r.col + .5) * 10 for r in df.itertuples()]; y = [tr[r.site].f - (r.row + .5) * 10 for r in df.itertuples()]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=32648)
    ll = g.to_crs(4326).geometry; df["lon"], df["lat"] = ll.x.round(6), ll.y.round(6); g["lon"], g["lat"] = df.lon, df.lat
    df.to_csv(OUT / "key/key.csv", index=False); g.to_parquet(OUT / "key/samples.parquet")
    (OUT / "RUBRIC.md").write_text(RUBRIC)
    import multiprocessing as mp
    jobs = [(r.site, r.sample_id, r.row, r.col, r.lon, r.lat) for r in df.itertuples()]
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(sheet, jobs, chunksize=2))
    print(df.groupby(["site", "stratum"]).size().unstack().to_string(), f"\n{len(df)} sheets -> {OUT / 'sheets'}")


def google_chip(lon, lat, sid, half_m=CHIP * 5):
    """Google satellite tiles at zoom 17 around the point, cropped to the S2 chip extent (cached)."""
    f = OUT / "google" / f"{sid}.png"
    import matplotlib.image as mpimg
    if f.exists():
        return mpimg.imread(f)
    from PIL import Image
    n = 2 ** ZOOM
    gx = (lon + 180) / 360 * n * 256
    gy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n * 256
    mpp = 156543.03392 * math.cos(math.radians(lat)) / n
    h = int(round(half_m / mpp))
    x0, x1, y0, y1 = int(gx - h), int(gx + h), int(gy - h), int(gy + h)
    img = Image.new("RGB", (x1 - x0, y1 - y0))
    for tx in range(x0 // 256, x1 // 256 + 1):
        for ty in range(y0 // 256, y1 // 256 + 1):
            url = f"https://mt{(tx + ty) % 4}.google.com/vt/lyrs=s&x={tx}&y={ty}&z={ZOOM}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                t = Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=20).read())).convert("RGB")
                img.paste(t, (tx * 256 - x0, ty * 256 - y0))
            except Exception:
                pass
    img.save(f)
    return np.asarray(img) / 255


def sheet(job):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    site, sid, row, col, lon, lat = job
    d = SP.site_dir(site) / "data"; ST = d / "stacks"; h = CHIP // 2
    w3 = Window(col - 1, row - 1, 3, 3)
    rd = lambda b, w, fill=0: rasterio.open(ST / f"S2_{b}.tif").read(window=w, boundless=True, fill_value=fill)  # noqa: E731
    with rasterio.open(ST / "S2_cs_cdf.tif") as r:
        dates = pd.to_datetime(list(r.descriptions))
    cs = rd("cs_cdf", w3, 65535).reshape(len(dates), -1)
    b8 = rd("B8", w3).reshape(len(dates), -1).astype("float32"); b4 = rd("B4", w3).reshape(len(dates), -1).astype("float32")
    clear = ((cs >= 60) & (cs <= 100)).mean(1) >= 0.78
    nd = (b8 - b4) / np.maximum(b8 + b4, 1)
    ok = clear & (dates >= S2_START) & (dates <= S2_END) & (b8.min(1) > 0)
    idx = list(np.flatnonzero(ok))
    octo = [i for i in idx if dates[i] >= pd.Timestamp("2026-10-01")]
    rest = [i for i in idx if i not in octo]
    k = max(0, 12 - len(octo))
    if len(rest) > k:
        rest = [rest[int(j)] for j in np.linspace(0, len(rest) - 1, k).round()] if k else []
    idx = sorted(rest + octo)[-12:]
    win = Window(col - h, row - h, CHIP, CHIP)
    L = {b: rasterio.open(ST / f"S2_{b}.tif").read([i + 1 for i in idx], window=win, boundless=True, fill_value=0).astype("float32")
         for b in ("B8", "B5", "B4", "B3", "B2")} if idx else {}
    st = lambda b, k: np.clip((L[b][k] - STRETCH[b][0]) / (STRETCH[b][1] - STRETCH[b][0]), 0, 1)  # noqa: E731
    with rasterio.open(d / "features/s1_15d_2026nov.tif") as r:
        names = list(r.descriptions)
        v = r.read(window=w3, boundless=True, fill_value=-32768).reshape(len(names), -1).astype("float32")
    v[v <= -32000] = np.nan; v /= 100
    s1 = pd.DataFrame({"track": ["_".join(n.split("_")[:2]) for n in names], "var": [n.split("_")[2] for n in names],
                       "mean": np.nanmean(v, 1), "median": np.nanmedian(v, 1)})
    s1["date"] = [pd.Timestamp(f"{n[-7:-3]}-{n[-3:-1]}-{'01' if n[-1] == 'a' else '16'}") + pd.Timedelta(days=7) for n in names]
    gimg = google_chip(lon, lat, sid)
    nc = 6; nr = max(1, math.ceil(len(idx) / nc))
    fig = plt.figure(figsize=(15, 2.5 * 2 * nr + 9.5))
    gs = fig.add_gridspec(2 * nr + 4, nc, height_ratios=[2.5] * (2 * nr) + [3.4, 2.2, 2.2, 2.2])
    for blk, (bands, lab) in enumerate(((("B5", "B3", "B2"), "5-3-2"), (("B8", "B4", "B3"), "8-4-3"))):
        for j in range(len(idx)):
            ax = fig.add_subplot(gs[blk * nr + j // nc, j % nc])
            ax.imshow(np.dstack([st(b, j) for b in bands]), interpolation="nearest")
            ax.add_patch(plt.Rectangle((h - 1.5, h - 1.5), 3, 3, fill=False, ec="yellow" if blk == 0 else "white", lw=1.4))
            ax.set_title(f"{lab}  {dates[idx[j]]:%Y-%m-%d}", fontsize=9); ax.set_axis_off()
    ax = fig.add_subplot(gs[2 * nr, 0:2]); ax.imshow(gimg)
    c = gimg.shape[0] / 2; b = gimg.shape[0] * 1.5 / CHIP
    ax.add_patch(plt.Rectangle((c - b, c - b), 2 * b, 2 * b, fill=False, ec="yellow", lw=1.4))
    ax.set_title("Google satellite (date unknown, context only)", fontsize=9); ax.set_axis_off()
    ax = fig.add_subplot(gs[2 * nr, 2:])
    dd = dates[ok]; ndo = nd[ok]
    ax.fill_between(dd, ndo.min(1), ndo.max(1), color="green", alpha=.15, label="3x3 min-max")
    ax.plot(dd, ndo.mean(1), "o-", color="green", ms=4, label="3x3 mean")
    ax.plot(dd, np.median(ndo, 1), "--", color="black", lw=1, label="3x3 median")
    ax.set_ylim(-0.1, 1); ax.set_xlim(S2_START, S2_END); ax.set_ylabel("NDVI (clear)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    for i, var in enumerate(("VH", "VV", "VHmVV")):
        ax = fig.add_subplot(gs[2 * nr + 1 + i, :])
        for t, g in s1[s1["var"] == var].groupby("track"):
            col_ = {"RO164_DES": "#1f77b4", "RO099_ASC": "#d62728"}.get(t)
            ax.plot(g.date, g["mean"], "o-", ms=3, color=col_, label=f"{t} 3x3 mean")
            ax.plot(g.date, g["median"], "--", lw=1, color=col_, alpha=.7, label=f"{t} median")
        ax.set_xlim(S2_START, S2_END); ax.set_ylabel(f"S1 {var.replace('m', '-')} (dB)"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=4)
    fig.suptitle(f"{sid}   (box = the 3 x 3 px block = 30 m; S2 chips 640 m wide; radar 15-day, Nov 2025 - Oct 2026)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(OUT / "sheets" / f"{sid}.png", dpi=60); plt.close(fig)


RUBRIC = """# Independent blind review: is this 3 x 3 block STANDING SUGARCANE (or cane-like cassava) in October 2026?

Each sheet = one 3 x 3 pixel block (30 m, the box in the centre of every chip). Blocks were chosen inside uniform areas,
so they are normally NOT on a field edge. Decide what stands there at the END of the 2026 rainy season (Oct 2026).

What the sheet shows
- S2 chips (640 m wide) on cloud-free dates Nov 2025 - Oct 2026, every October date included:
  * 5-3-2 (B5 red-edge / green / blue): crops separate well here; sugarcane canopy has its own tone and a coarse,
    uniform texture over a large block, cassava is usually patchier / less dense, rice paddies show small bunded plots.
  * 8-4-3 NIR false colour (B8/B4/B3): bright red = dense green vegetation, pink = sparse/young, cyan/grey/brown =
    bare soil / stubble / built-up, dark blue/black = water.
- Google satellite chip: high resolution but DATE UNKNOWN (may be years old). Use it for context only (orchard rows,
  forest, buildings, paddy bunds, field size), never as proof of the 2026 crop.
- NDVI of the block on clear dates: 3 x 3 mean (line), median (dashed), min-max band (a wide band = mixed block).
- Sentinel-1 radar 15-day composites Nov 2025 - Oct 2026 (VH, VV, VH-VV; two orbits; 3 x 3 mean and median).
  Higher VH = more vegetation volume; a deep VH drop = harvest / bare field. Radar sees through clouds (Jun - Oct).

Typical patterns (NE the country; 2026 had a drought in May - June, regrowth may be slow).
- SUGARCANE: harvested Dec 2025 - Apr 2026 (bare, NDVI ~0.2-0.35, VH ~ -20 dB), ratoon regrowth from Mar - May, tall
  dense canopy Jul - Oct (VH high and stable ~ -15 / -16 dB). Plant cane planted Oct - Dec 2025 grows through the dry
  season without an early 2026 dip. A cane field can show a weak harvest dip (trash left on the field).
- CASSAVA: less dense canopy, peak NDVI ~0.6-0.75, planted Mar - May or Oct - Nov, harvested at scattered times.
  Counts as cane-like: answer "cane" and put cassava in "what".
- RICE: bare/dry Jan - May with no regrowth, flooded dark plots Jun - Aug (VH very low), green Aug - Oct.
- FALLOW / GRASS after the 2026 harvest: low NDVI and low VH through Jun - Oct.
- TREES / ORCHARD / EUCALYPTUS / RUBBER / FOREST: high all year, no harvest dip.
- WATER, BUILT-UP, BARE: obvious.

Answer one row per sheet: sample_id,answer,what,confidence,note
- answer: cane | not_cane | unsure
- what: sugarcane | cassava | rice | other_crop | tree | grass_fallow | water | built | bare | mixed_edge | unknown
- confidence: high | medium | low ;  note: a few words. Judge only from the sheet; every block is independent.
Say not_cane when the evidence clearly shows another land cover; do not lean to cane by default, and do not lean to
not_cane by default either.
"""


def compare():
    key = pd.read_csv(OUT / "key/key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((OUT / "labels").glob("batch_*.csv"))], ignore_index=True)
    d = key.merge(lab[["sample_id", "answer", "what", "confidence", "note"]], on="sample_id", how="left")
    d["y"] = d.answer.map({"cane": 1, "not_cane": 0}); d["yh"] = (d.stratum == "map_cane").astype(int)
    rows = []
    for site in AOIS + ["ALL"]:
        x = d[((d.site == site) | (site == "ALL")) & d.y.notna()]
        y, yh, w = x.y.astype(int), x.yh, x.weight
        tp, fp, fn = (w * (yh & y)).sum(), (w * (yh & (1 - y))).sum(), (w * ((1 - yh) & y)).sum()
        P, R = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
        ac = 100 / 4046.8564224
        rows.append({"site": site, "n_sure": len(x), "n_unsure": int(((d.site == site) | (site == "ALL")).sum() - len(x)),
                     "precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / max(P + R, 1e-9), 3),
                     "oa": round(float((w * (yh == y)).sum() / w.sum()), 3),
                     "map_cane_ac": round(float(x[x.yh == 1].groupby("site").n_stratum_px.first().sum() * ac)),
                     "est_true_cane_ac": round(float((w * y).sum() * ac))})
    r = pd.DataFrame(rows)
    d.to_csv(OUT / "review_results.csv", index=False); r.to_csv(OUT / "scores.csv", index=False)
    print(d.groupby("stratum").answer.value_counts().unstack().to_string())
    print(pd.crosstab(d.stratum, d.what).to_string())
    print(r.to_string(index=False))


if __name__ == "__main__":
    {"sample": sample, "compare": compare}[sys.argv[1]]()

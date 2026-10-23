"""Label QC of the 2025 cane labels: find pixels whose label is probably wrong, so training can skip them.

Dropping edge pixels is not enough: big label polygons also hold small non-cane patches inside (and the non-cane
side holds unmapped cane). Flagged pixels are EXCLUDED (label 255 in the clean raster), never relabelled.

Inputs : data/labels/cane2025_labels.tif (label, poly_id, edge_px), data/features/s1_15d_2025.tif (dB x 100),
         data/stacks/S2_NDVI.tif + S2_CLEAR.tif (QC only, never a model feature)
Outputs: data/labels/cane2025_qc.tif, bands:
           1 label_clean   1 cane, 0 non-cane, 255 excluded (flagged) or outside the AOI
           2 flags         bit mask, see FLAGS below (0 = clean)
           3 block_id      500 m grid block (50 x 50 px): split unit for big polygons, spatial CV and the test set
           4 unit_id       cane polygon x 500 m block (0 = non-cane)
           5 p_oof         out-of-fold cane probability x 100 (LightGBM, 2 km block folds, all 162 S1 bands)
           6 d_unit        RMS distance (dB x 100) of the pixel's VH curve to the median curve of its unit (cane only)
           7 edge_px       copied from the label raster
         data/labels/cane2025_qc_summary.json, reports/label_qc/ (flag maps, curves per flag, stats)
Stages are cached in data/labels/qc_cache/ (metrics.npz); `--flags` reruns only the flag rules (seconds).
usage: python scripts/label_qc.py [--flags]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402  (GDAL cache cap, all CPUs)

import json, os, time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import site_paths as SP

PROJ = SP.site_dir(os.environ["SC_ROOT"]) if os.environ.get("SC_ROOT") else Path(__file__).resolve().parents[1]   # SC_ROOT = another site
FEATURES = os.environ.get("QC_FEATURES", "s1_15d_2025.tif")                      # sites: s1_15d_2025jan.tif
LAB = PROJ / "data/labels"
CACHE = LAB / "qc_cache"
REP = PROJ / "reports/label_qc"
BLOCK, CV_BLOCK, MIN_UNIT_PX = 50, 200, 25      # 500 m units, 2 km CV blocks
N_TRAIN, FOLDS, SEED, ROWS = 300_000, 5, 0, 256
NCPU = os.cpu_count() or 8

FLAGS = {  # bit: (name, rule) -- thresholds in apply_flags()
    1: "outlier_unit",     # cane: VH curve far from the median curve of its polygon x 500 m unit, and p_oof < 0.5
    2: "water",            # cane: VH < -22 dB on >= 60 % of half-months
    4: "built",            # cane: median VV > -6 dB and stable (std < 1.5 dB)
    8: "tree",             # cane: no dip all season (VH min > -16 dB) and flat (p90 - p10 < 2.5 dB)
    16: "bare",            # cane: never vegetated (VH p90 < -19 dB)
    32: "s2_not_green",    # cane: >= 2 clear S2 dates Aug-Oct 2025 and all NDVI < 0.40
    64: "unit_bad",        # cane: >= 50 % of the unit flagged by the rules above -> whole unit excluded
    128: "cl_cane_low",    # cane label, out-of-fold p < 0.10
    256: "cl_noncane_high",  # non-cane label, out-of-fold p > 0.90 (likely unmapped cane)
}


def read_labels():
    with rasterio.open(LAB / "cane2025_labels.tif") as r:
        return r.read(1).astype("uint8"), r.read(2), r.read(3).astype("uint8"), r.profile.copy()


def band_groups(names):
    g = {}
    for i, n in enumerate(names):
        t = n.split("_")
        g.setdefault(t[2], []).append(i)   # VV / VH / VHmVV, both tracks
    return g


def stage_metrics(lab, pid, edge):
    """One-time heavy part: unit medians + distances, physics stats, S2 stats, OOF probabilities."""
    t0 = time.time()
    H, W = lab.shape
    rr, cc = np.indices((H, W), dtype="int32")
    block = (rr // BLOCK) * ((W + BLOCK - 1) // BLOCK) + cc // BLOCK
    cvb = (rr // CV_BLOCK) * ((W + CV_BLOCK - 1) // CV_BLOCK) + cc // CV_BLOCK
    del rr, cc
    rng = np.random.default_rng(SEED)
    fold_of_cvb = rng.integers(0, FOLDS, cvb.max() + 1)
    fold = fold_of_cvb[cvb].astype("int8")
    cane = lab == 1
    labelled = lab != 255
    # unit = cane polygon x 500 m block (ids dense, 0 = non-cane)
    key = np.where(cane, pid.astype("int64") * (block.max() + 1) + block, -1)
    _, unit_flat = np.unique(key.ravel(), return_inverse=True)
    unit = unit_flat.reshape(H, W).astype("int32")       # key -1 -> 0 (sorted first), cane units 1..
    print(f"units {unit.max():,} ({time.time() - t0:.0f} s)", flush=True)

    fpath = PROJ / "data/features" / FEATURES
    with rasterio.open(fpath) as r:
        names = list(r.descriptions)
    g = band_groups(names)
    vh_i, vv_i = g["VH"], g["VV"]

    # --- physics + unit distance need VH/VV of cane pixels: read the bands once each
    ci = np.flatnonzero(cane.ravel())
    with rasterio.open(fpath) as r:
        VH = np.stack([r.read(i + 1).ravel()[ci] for i in vh_i], 1).astype("float32") / 100
        VV = np.stack([r.read(i + 1).ravel()[ci] for i in vv_i], 1).astype("float32") / 100
    print(f"cane px {len(ci):,}, VH/VV read ({time.time() - t0:.0f} s)", flush=True)
    u = unit.ravel()[ci]
    p_ = pid.ravel()[ci]
    n_u = np.bincount(u)
    # median curve per unit; units below MIN_UNIT_PX fall back to the polygon, then the global cane median
    df = pd.DataFrame(VH); df["u"] = u
    med_u = df.groupby("u").median()
    df["p"] = p_
    med_p = df.drop(columns="u").groupby("p").median()
    ref = med_u.reindex(u).values
    small = n_u[u] < MIN_UNIT_PX
    n_p = np.bincount(p_)
    ref[small] = med_p.reindex(p_[small]).values
    tiny = small & (n_p[p_] < MIN_UNIT_PX)
    ref[tiny] = np.median(VH, 0)
    d_unit = np.sqrt(np.mean((VH - ref) ** 2, 1))
    del df, ref
    phys = {
        "vh_frac_lt22": (VH < -22).mean(1), "vh_min": VH.min(1), "vh_p90": np.percentile(VH, 90, 1),
        "vh_p10": np.percentile(VH, 10, 1), "vv_med": np.median(VV, 1), "vv_std": VV.std(1),
    }
    del VH, VV
    print(f"unit distance + physics ({time.time() - t0:.0f} s)", flush=True)

    # --- S2 NDVI on clear dates Aug-Oct 2025 (cane pixels)
    st = PROJ / "data/stacks"
    if (st / "S2_NDVI.tif").exists():       # first AOI: precomputed NDVI + CLEAR mask
        dates = pd.to_datetime(list(rasterio.open(st / "S2_NDVI.tif").descriptions))
        def read_nd_clr(i):
            with rasterio.open(st / "S2_NDVI.tif") as rn, rasterio.open(st / "S2_CLEAR.tif") as rc:
                return rn.read(i + 1).ravel()[ci].astype("float32") / 10000, rc.read(i + 1).ravel()[ci] == 1
    else:                                   # test sites: only B4 / B8 + Cloud Score+ (clear = cs_cdf >= 0.60)
        dates = pd.to_datetime(list(rasterio.open(st / "S2_B8.tif").descriptions))
        def read_nd_clr(i):
            with rasterio.open(st / "S2_B8.tif") as r8, rasterio.open(st / "S2_B4.tif") as r4, rasterio.open(st / "S2_cs_cdf.tif") as rc:
                b8, b4 = r8.read(i + 1).ravel()[ci].astype("float32"), r4.read(i + 1).ravel()[ci].astype("float32")
                cs = rc.read(i + 1).ravel()[ci]
                return (b8 - b4) / np.maximum(b8 + b4, 1), (cs >= 60) & (cs <= 100) & (b8 > 0)
    yr = int(os.environ.get("QC_S2_YEAR", "2025"))   # 2026 run: Aug - Oct 2026
    late = [i for i, d in enumerate(dates) if pd.Timestamp(f"{yr}-08-01") <= d <= pd.Timestamp(f"{yr}-10-31")]
    nd_max = np.full(len(ci), -1.0, "float32"); n_clear = np.zeros(len(ci), "int16")
    for i in late:
        nd, clr = read_nd_clr(i)
        n_clear += clr
        nd_max = np.where(clr, np.maximum(nd_max, nd), nd_max)
    print(f"S2 late NDVI ({len(late)} dates) ({time.time() - t0:.0f} s)", flush=True)

    # --- OOF probabilities: train on a sample of interior pixels, predict every labelled pixel with its fold's model
    import lightgbm as lgb
    cand = np.flatnonzero((labelled & (edge >= 2)).ravel())
    yall = lab.ravel()
    take = []
    for cls in (1, 0):
        c = cand[yall[cand] == cls]
        take.append(rng.choice(c, min(N_TRAIN // 2, len(c)), replace=False))
    take = np.sort(np.concatenate(take))
    with rasterio.open(fpath) as r:
        Xs = np.stack([r.read(i + 1).ravel()[take] for i in range(len(names))], 1).astype("float32")
    ys, fs = yall[take], fold.ravel()[take]
    ok = (Xs > -32000).all(1)
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, n_jobs=NCPU, verbose=-1, random_state=SEED)
    models, oof_auc = [], []
    from sklearn.metrics import roc_auc_score
    for k in range(FOLDS):
        tr, te = ok & (fs != k), ok & (fs == k)
        m = lgb.LGBMClassifier(**params).fit(Xs[tr], ys[tr])
        models.append(m)
        oof_auc.append(float(roc_auc_score(ys[te], m.predict_proba(Xs[te])[:, 1])))
        print(f"  fold {k}: train {tr.sum():,} test {te.sum():,} AUC {oof_auc[-1]:.3f} ({time.time() - t0:.0f} s)", flush=True)
    del Xs
    p_oof = np.full(H * W, 255, "uint8")
    with rasterio.open(fpath) as r:
        for r0 in range(0, H, ROWS):
            h = min(ROWS, H - r0)
            X = r.read(window=((r0, r0 + h), (0, W))).reshape(len(names), -1).T.astype("float32")
            sl = slice(r0 * W, (r0 + h) * W)
            lb, fd = labelled.ravel()[sl], fold.ravel()[sl]
            good = lb & (X > -32000).all(1)
            out = p_oof[sl]
            for k in range(FOLDS):
                s = good & (fd == k)
                if s.any():
                    out[s] = np.round(100 * models[k].predict_proba(X[s])[:, 1]).astype("uint8")
    print(f"OOF prediction of all labelled px ({time.time() - t0:.0f} s)", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "metrics.npz", ci=ci, d_unit=d_unit, nd_max=nd_max, n_clear=n_clear, unit=unit, block=block,
             p_oof=p_oof.reshape(H, W), oof_auc=np.array(oof_auc), **phys)


def apply_flags(lab, pid, edge, prof):
    t0 = time.time()
    m = np.load(CACHE / "metrics.npz")
    H, W = lab.shape
    ci, unit, block, p_oof = m["ci"], m["unit"], m["block"], m["p_oof"]
    fl = np.zeros(len(ci), "uint16")
    d = m["d_unit"]
    u = unit.ravel()[ci]
    # outlier: robust z within the unit (median + 3.5 MAD of the unit's distances) AND an absolute floor of 2.5 dB
    du = pd.Series(d).groupby(u)
    med = du.median().reindex(u).values
    mad = (pd.Series(np.abs(d - med)).groupby(u).median().reindex(u).values * 1.4826)
    # ... and only when the model also doubts the pixel (p_oof < 0.5): a unit-outlier with high p is mostly real cane
    # of a different calendar, e.g. plant cane planted Oct-Nov 2024 inside a ratoon unit (blind review + user, 12 Oct)
    pc = p_oof.ravel()[ci]
    fl |= np.where((d > med + 3.5 * np.maximum(mad, 0.3)) & (d > 2.5) & (pc < 50), 1, 0).astype("uint16")
    fl |= np.where(m["vh_frac_lt22"] >= 0.6, 2, 0).astype("uint16")
    fl |= np.where((m["vv_med"] > -6) & (m["vv_std"] < 1.5), 4, 0).astype("uint16")
    fl |= np.where((m["vh_min"] > -16) & (m["vh_p90"] - m["vh_p10"] < 2.5), 8, 0).astype("uint16")
    fl |= np.where(m["vh_p90"] < -19, 16, 0).astype("uint16")
    fl |= np.where((m["n_clear"] >= 2) & (m["nd_max"] < 0.40), 32, 0).astype("uint16")
    share = pd.Series((fl & 63) > 0).groupby(u).mean().reindex(u).values
    nu = np.bincount(u)[u]
    fl |= np.where((share >= 0.5) & (nu >= 10), 64, 0).astype("uint16")
    flags = np.zeros(H * W, "uint16")
    flags[ci] = fl
    lf = lab.ravel()
    pf = p_oof.ravel()
    flags |= np.where((lf == 1) & (pf < 10), 128, 0).astype("uint16")
    flags |= np.where((lf == 0) & (pf > 90) & (pf != 255), 256, 0).astype("uint16")
    flags = flags.reshape(H, W)
    clean = np.where((lab != 255) & (flags == 0), lab, 255).astype("uint8")
    dmap = np.zeros(H * W, "float32"); dmap[ci] = d
    dmap = np.round(dmap * 100).clip(0, 32767).reshape(H, W)
    p = dict(prof, count=7, dtype="int32", nodata=None, compress="deflate", predictor=2)
    with rasterio.open(LAB / "cane2025_qc.tif", "w", **p) as dst:
        for i, (a, n) in enumerate([(clean, "label_clean"), (flags, "flags"), (block, "block_id"), (unit, "unit_id"),
                                    (p_oof, "p_oof"), (dmap, "d_unit"), (edge, "edge_px")], 1):
            dst.write(a.astype("int32"), i); dst.set_band_description(i, n)
    ac = 100 / 4046.856
    n_c, n_n = int((lab == 1).sum()), int((lab == 0).sum())
    stats = {}
    for b, name in FLAGS.items():
        on = (flags & b) > 0
        stats[name] = {"px": int(on.sum()), "acres": round(on.sum() * ac), "pct_of_cane_label": round(100 * (on & (lab == 1)).sum() / n_c, 2),
                       "pct_of_noncane_label": round(100 * (on & (lab == 0)).sum() / n_n, 2),
                       "only_this_flag_px": int((flags == b).sum())}
    s = {"oof_auc_per_fold": [round(float(a), 3) for a in m["oof_auc"]],
         "cane_label_px": n_c, "noncane_label_px": n_n,
         "cane_excluded_pct": round(100 * ((lab == 1) & (flags > 0)).sum() / n_c, 2),
         "noncane_excluded_pct": round(100 * ((lab == 0) & (flags > 0)).sum() / n_n, 2),
         "clean_cane_px": int((clean == 1).sum()), "clean_noncane_px": int((clean == 0).sum()),
         "clean_cane_interior_px": int(((clean == 1) & (edge >= 2)).sum()),
         "clean_noncane_interior_px": int(((clean == 0) & (edge >= 2)).sum()),
         "units": int(unit.max()), "blocks_500m": int(len(np.unique(block[lab != 255]))), "flags": stats}
    (LAB / "cane2025_qc_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1)); print(f"flags ({time.time() - t0:.0f} s)", flush=True)
    report(lab, flags, p_oof, m)


def report(lab, flags, p_oof, m):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    REP.mkdir(parents=True, exist_ok=True)
    # flag map (downsampled 4x for the png)
    rgb = np.full(lab.shape + (3,), 1.0)
    rgb[lab == 1] = (0.75, 0.93, 0.72); rgb[lab == 0] = (0.92, 0.92, 0.92)
    for b, col in ((1, (1, .5, 0)), (2, (0, 0, 1)), (4, (.9, .6, 0)), (8, (0, .4, 0)), (16, (.6, .4, .2)),
                   (32, (.8, 0, .8)), (64, (1, 0, 0)), (128, (.6, 0, 0)), (256, (0, .6, 1))):
        rgb[(flags & b) > 0] = col
    fig, ax = plt.subplots(figsize=(12, 10.5))
    ax.imshow(rgb[::3, ::3], interpolation="nearest"); ax.set_axis_off()
    from matplotlib.patches import Patch
    leg = [Patch(color=(0.75, 0.93, 0.72), label="cane label (clean)"), Patch(color=(0.92, 0.92, 0.92), label="non-cane (clean)")]
    leg += [Patch(color=c, label=FLAGS[b]) for b, c in ((1, (1, .5, 0)), (2, (0, 0, 1)), (4, (.9, .6, 0)), (8, (0, .4, 0)),
                                                       (16, (.6, .4, .2)), (32, (.8, 0, .8)), (64, (1, 0, 0)), (128, (.6, 0, 0)), (256, (0, .6, 1)))]
    ax.legend(handles=leg, loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(REP / "flag_map.png", dpi=80); plt.close(fig)
    # histograms of the OOF probability per label
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, cls, t in ((ax[0], 1, "cane label"), (ax[1], 0, "non-cane label")):
        v = p_oof[(lab == cls) & (p_oof != 255)]
        a.hist(v, bins=50, color="green" if cls else "grey"); a.set_title(f"{t}: out-of-fold p(cane) x 100"); a.set_yscale("log")
    fig.tight_layout(); fig.savefig(REP / "p_oof_hist.png", dpi=80); plt.close(fig)


def main():
    lab, pid, edge, prof = read_labels()
    if "--flags" not in sys.argv or not (CACHE / "metrics.npz").exists():
        stage_metrics(lab, pid, edge)
    apply_flags(lab, pid, edge, prof)


if __name__ == "__main__":
    main()

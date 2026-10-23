"""Exploratory look at the 2025 labels against the Sentinel-1 15-day composites (before any model).

Inputs : data/features/s1_15d_2025.tif, data/labels/cane2025_labels.tif, data/stacks/S2_CLEAR.tif (+ S2_NDVI.tif)
Outputs (reports/eda/):
  profiles.png            median and 25-75 % band of VV, VH, VH-VV per half-month, cane vs non-cane, both tracks
  cane_clusters.png       k-means groups inside the cane label (VH, both tracks): what kinds of "cane" there are
  band_auc.csv / .png     how well each single band separates cane from non-cane (ROC AUC, 0.5 = useless)
  s2_clear_by_month.csv   share of the AOI with a cloud-free S2 observation per month
  summary.json
Pixels: interior only (>= 2 px from the cane / non-cane border), 40,000 per class, random, fixed seed.
usage: python scripts/eda.py
"""
from __future__ import annotations

import os
os.environ.setdefault("GDAL_CACHEMAX", "256")
os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "reports/eda"
N_PER_CLASS, SEED, K = 40_000, 0, 6


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import roc_auc_score
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    with rasterio.open(PROJ / "data/labels/cane2025_labels.tif") as r:
        lab, pid, edge = r.read(1), r.read(2), r.read(3)
    rows, cols, y = [], [], []
    for cls in (1, 0):
        rr, cc = np.nonzero((lab == cls) & (edge >= 2))
        k = rng.choice(len(rr), min(N_PER_CLASS, len(rr)), replace=False)
        rows.append(rr[k]); cols.append(cc[k]); y.append(np.full(len(k), cls))
    rr, cc, y = np.concatenate(rows), np.concatenate(cols), np.concatenate(y)
    with rasterio.open(PROJ / "data/features/s1_15d_2025.tif") as r:
        names = list(r.descriptions)
        X = np.stack([r.read(i + 1)[rr, cc] for i in range(len(names))], 1)
    ok = np.isfinite(X).all(1); X, y, rr, cc = X[ok], y[ok], rr[ok], cc[ok]
    df = pd.DataFrame(X, columns=names); df["y"] = y
    print(f"pixels {len(df)} (cane {int(y.sum())}) ({time.time() - t0:.0f} s)", flush=True)
    periods = sorted({n.split("_")[-1] for n in names})
    tracks = sorted({"_".join(n.split("_")[:2]) for n in names})

    # 1. profiles
    fig, ax = plt.subplots(3, len(tracks), figsize=(9 * len(tracks), 12), squeeze=False)
    for j, t in enumerate(tracks):
        for i, v in enumerate(("VV", "VH", "VHmVV")):
            a = ax[i, j]
            for cls, col, lb in ((1, "green", "cane (2025 label)"), (0, "grey", "non-cane")):
                q = df[df.y == cls][[f"{t}_{v}_{p}" for p in periods]].quantile([.25, .5, .75]).values
                a.plot(range(len(periods)), q[1], color=col, lw=2, label=lb); a.fill_between(range(len(periods)), q[0], q[2], color=col, alpha=.2)
            a.set_title(f"{t} {v}"); a.set_xticks(range(0, len(periods), 2)); a.set_xticklabels(periods[::2], rotation=90, fontsize=7)
            a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "profiles.png", dpi=70); plt.close(fig)

    # 2. groups inside the cane label (VH both tracks, z-scored per band)
    vh = [n for n in names if "_VH_" in n]
    C = df[df.y == 1][vh].values; Cz = (C - C.mean(0)) / (C.std(0) + 1e-9)
    km = KMeans(K, n_init=5, random_state=SEED).fit(Cz)
    lab_k = km.labels_; share = np.bincount(lab_k, minlength=K) / len(lab_k)
    non_med = df[df.y == 0][vh].median().values
    fig, ax = plt.subplots(1, len(tracks), figsize=(9 * len(tracks), 5), squeeze=False)
    for j, t in enumerate(tracks):
        cols_t = [k for k, n in enumerate(vh) if n.startswith(t)]
        for k in np.argsort(-share):
            ax[0, j].plot(range(len(periods)), np.median(C[lab_k == k][:, cols_t], 0), lw=2, label=f"group {k} ({share[k]:.0%})")
        ax[0, j].plot(range(len(periods)), non_med[cols_t], "k--", lw=1.5, label="non-cane median")
        ax[0, j].set_title(f"cane label, k-means groups, VH {t}"); ax[0, j].set_xticks(range(0, len(periods), 2))
        ax[0, j].set_xticklabels(periods[::2], rotation=90, fontsize=7); ax[0, j].grid(alpha=.3); ax[0, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "cane_clusters.png", dpi=70); plt.close(fig)

    # 3. single-band AUC
    auc = pd.Series({n: roc_auc_score(df.y, df[n]) for n in names}).rename("auc").to_frame()
    auc["sep"] = (auc.auc - 0.5).abs() + 0.5
    auc.to_csv(OUT / "band_auc.csv")
    fig, ax = plt.subplots(figsize=(16, 5))
    for t in tracks:
        for v, ls in (("VV", ":"), ("VH", "-"), ("VHmVV", "--")):
            ax.plot(range(len(periods)), [auc.loc[f"{t}_{v}_{p}", "sep"] for p in periods], ls, label=f"{t} {v}")
    ax.set_xticks(range(len(periods))); ax.set_xticklabels(periods, rotation=90, fontsize=7); ax.set_ylabel("separability (|AUC-0.5|+0.5)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "band_auc.png", dpi=70); plt.close(fig)

    # 4. S2 clear share per month (AOI)
    with rasterio.open(PROJ / "data/stacks/S2_CLEAR.tif") as r:
        d = pd.to_datetime(list(r.descriptions)); inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
        clr = pd.Series({d[i]: float((r.read(i + 1)[inside] == 1).mean()) for i in range(r.count)})
    m = clr.groupby(clr.index.to_period("M")).agg(["count", "max", "mean"]).rename(columns={"count": "dates", "max": "best_date_clear", "mean": "mean_clear"})
    m.round(3).to_csv(OUT / "s2_clear_by_month.csv")
    s = {"pixels": int(len(df)), "cane": int(y.sum()), "kmeans_group_share": {int(k): float(v) for k, v in enumerate(share)},
         "top_bands_auc": auc.sort_values("sep", ascending=False).head(15).round(3).to_dict()["sep"],
         "s2_clear_by_month": m.round(2).reset_index().astype(str).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("kmeans_group_share", "top_bands_auc")}, indent=1)); print(m.round(2).to_string())
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

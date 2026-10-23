"""Which pixels does the model lose when the series starts in Jan 2025 instead of Nov 2024? (out-of-fold, same folds)

Inputs : data/model/m2025/samples.npz (train sample, from model_2025.py), the feature stack (only to rebuild the
         sample's pixel positions) and the QC raster
Outputs: reports/window/window_diff.csv  every train px whose OOF class differs between the two windows (+ pixel_id)
         reports/window/curves.png       median VH / VV curves: flipped groups vs stable cane / non-cane
         reports/window/summary.json
usage: python scripts/window_diff.py [startA startB]     (default 202411a 202501a)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

import model_2025 as M  # noqa: E402

OUT = M.PROJ / "reports/window"


def train_pixel_ids():
    """Rebuild the train sample's flat pixel indices exactly as model_2025.samples() drew them."""
    names, X, clean, raw, block, edge, _ = M.load_all()
    cvb, test, fold = M.split(clean.shape)
    rng = np.random.default_rng(M.SEED)
    ok = (X.reshape(len(names), -1) > -32000).all(0)
    idx = []
    for cls in (1, 0):
        c = np.flatnonzero((clean.ravel() == cls) & (edge.ravel() >= 2) & ~test.ravel() & ok)
        idx.append(rng.choice(c, M.N_TRAIN // 2, replace=False))
    return np.sort(np.concatenate(idx)), clean.shape[1]


def main():
    a, b = (sys.argv[1:3] if len(sys.argv) >= 3 else ("202411a", "202501a"))
    OUT.mkdir(parents=True, exist_ok=True)
    d = M.samples()
    names, y, folds = list(d["names"]), d["ytr"], d["ftr"]
    pid, W = train_pixel_ids()
    assert len(pid) == len(y)
    P = {}
    for st in (a, b):
        F, _ = M.build(d["Xtr"], names, f"VVVH@{st}")
        P[st], m = M.cv("lgbm", F, y, folds)
        M.log(f"{st}: F1 {m['f1']:.4f}")
    pa, pb = P[a], P[b]
    ca, cb = pa >= 0.5, pb >= 0.5
    grp = np.select([(y == 1) & ca & ~cb, (y == 0) & ~ca & cb, (y == 1) & ~ca & cb, (y == 0) & ca & ~cb,
                     (y == 1) & ca & cb, (y == 0) & ~ca & ~cb],
                    [f"cane_lost_{b}", f"noncane_falsecane_{b}", f"cane_gained_{b}", f"noncane_fixed_{b}",
                     "cane_ok_both", "noncane_ok_both"], "wrong_both")
    rows, cols = np.divmod(pid, W)
    df = pd.DataFrame({"pixel_id": pid, "row": rows, "col": cols, "label": y, f"p_{a}": pa.round(3), f"p_{b}": pb.round(3), "group": grp})
    # VH Sep 2024 - Feb 2025 mean of both tracks, for a quick read of each pixel
    p = M.parse(names)
    for per in sorted(p.period.unique()):
        if per <= "202503b":
            ii = p.index[(p["var"] == "VH") & (p.period == per)].values
            df[f"VH_{per}"] = (d["Xtr"][:, ii].astype("float32") / 100).mean(1).round(1)
    flips = df[~df.group.isin(["cane_ok_both", "noncane_ok_both", "wrong_both"])]
    flips.to_csv(OUT / "window_diff.csv", index=False)
    s = {"F1": {a: float(M.metrics(y, pa)["f1"]), b: float(M.metrics(y, pb)["f1"])},
         "groups": df.group.value_counts().to_dict(),
         "acres_per_100k_train_px_note": "train sample is balanced 150k / 150k; counts are sample pixels, not map area"}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps(s, indent=1, default=float))
    # curves: median per period, VH and VV (track mean), each group
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pers = sorted(p.period.unique())
    fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for k, var in enumerate(("VH", "VV")):
        V = np.stack([(d["Xtr"][:, p.index[(p["var"] == var) & (p.period == q)].values].astype("float32") / 100).mean(1) for q in pers], 1)
        for g, col, ls in (("cane_ok_both", "green", "-"), ("noncane_ok_both", "grey", "-"), (f"cane_lost_{b}", "red", "-"),
                           (f"noncane_falsecane_{b}", "orange", "-"), (f"cane_gained_{b}", "blue", "--"), (f"noncane_fixed_{b}", "purple", "--")):
            m = df.group.values == g
            if m.sum() >= 20:
                ax[k].plot(range(len(pers)), np.median(V[m], 0), ls, color=col, lw=2, label=f"{g} ({m.sum():,})")
        ax[k].axvspan(pers.index(a) - 0.5, pers.index(b) - 0.5, color="gold", alpha=0.15)
        ax[k].set_ylabel(f"{var} (dB, median)"); ax[k].grid(alpha=.3); ax[k].legend(fontsize=8, ncol=2)
    ax[1].set_xticks(range(len(pers))); ax[1].set_xticklabels(pers, rotation=90, fontsize=8)
    ax[0].set_title(f"Train px whose out-of-fold class changes when the series starts at {b} instead of {a} (gold = dropped months)")
    fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=70)
    M.log("done")


if __name__ == "__main__":
    main()

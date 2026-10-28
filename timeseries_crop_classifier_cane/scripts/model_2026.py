"""Standing-cane model for the 2026 season: 2025 training sites + the first AOI's 2026 data with QC-cleaned labels.

Why: the 2025-only model called much 2026 cane non-cane (2026 drought: shallower harvest dip, weaker regrowth); a
blind review of its "lost" fields found ~73 % still cane. 2026 has no labels, so the 2025 cane map is used on the
2026 data AFTER label QC on the 2026 data (label_qc.py with SC_ROOT=sites/site119_2026, QC_FEATURES=
s1_15d_2026jan.tif, QC_S2_YEAR=2026): pixels whose 2026 signal disagrees with the 2025 label are excluded.
Honest check: the 2 km blocks holding the 80 blind-reviewed 2026 pixels are left out of the 2026 training data
(and of the 2025 data of the same AOI), so those pixels are a held-out test.
Train: site119 (2025), site21 (2025), site01 (2025), site119_2026 (2026, band names shifted
2026 -> 2025); 100k cane + 100k non-cane per site (clean, edge >= 2); LightGBM defaults; VVVH Jan - 10 Oct.
Output: data/model/m2026/model.joblib (+ cv.json: leave-one-site-out and held-out review pixels)
usage: python scripts/model_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

import model_2025 as M  # noqa: E402
import site_paths as SP

PROJ = M.PROJ
OUT = PROJ / "data/model/m2026"
FS, N_PER_CLASS, SEED = "VVVH@202501a", 100_000, 0
SITES = {  # name: (root, features file, shift 2026 -> 2025)
    "site119": (PROJ, PROJ / "data/features/s1_15d_2025.tif", False),
    "site21": (PROJ / "sites/site21", PROJ / "sites/site21/data/features/s1_15d_2025jan.tif", False),
    "site01": (PROJ / "sites/site01", PROJ / "sites/site01/data/features/s1_15d_2025jan.tif", False),
    "site119_2026": (SP.site_dir("site119_2026"), SP.site_dir("site119_2026") / "data/features/s1_15d_2026jan.tif", True),
}
REVIEW = SP.site_dir("site119_2026") / "review_review2026/review_results.csv"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def shift(names):
    return [n[:-7] + str(int(n[-7:-3]) - 1) + n[-3:] for n in names]


def blocks(rr, cc, W):
    return (rr // M.CV_BLOCK) * ((W + M.CV_BLOCK - 1) // M.CV_BLOCK) + cc // M.CV_BLOCK


def load(name, k, held_blocks):
    root, feat, sh = SITES[name]
    with rasterio.open(root / "data/labels/cane2025_qc.tif") as r:
        clean, edge = r.read(1), r.read(7)
    with rasterio.open(feat) as r:
        names, X = list(r.descriptions), r.read()
    names = shift(names) if sh else names
    H, W = clean.shape
    Xf = X.reshape(len(names), -1)
    rr, cc = np.divmod(np.arange(H * W), W)
    blk = blocks(rr, cc, W)
    ok = (Xf > -32000).all(0)
    if name.startswith("site119"):
        ok &= ~np.isin(blk, list(held_blocks))
    rng = np.random.default_rng(SEED + k)
    idx = []
    for cls in (1, 0):
        c = np.flatnonzero((clean.ravel() == cls) & (edge.ravel() >= 2) & ok)
        idx.append(rng.choice(c, min(N_PER_CLASS, len(c)), replace=False))
    idx = np.sort(np.concatenate(idx))
    F, cols = M.build(Xf[:, idx].T, names, FS)
    log(f"{name}: {len(idx):,} px, {len(cols)} features")
    return F, clean.ravel()[idx].astype(int), cols


def review_features(W=4019):
    d = pd.read_csv(REVIEW)
    root, feat, _ = SITES["site119_2026"]
    with rasterio.open(feat) as r:
        names = shift(list(r.descriptions))
        X = np.stack([r.read(i + 1)[d.row.values, d.col.values] for i in range(r.count)], 1)
    F, cols = M.build(X, names, FS)
    F[F <= -320] = np.nan
    return d, F, cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d_rev = pd.read_csv(REVIEW)
    held = set(blocks(d_rev.row.values, d_rev.col.values, 4019).tolist())
    log(f"held-out 2 km blocks (review px): {len(held)}")
    parts = {s: load(s, k, held) for k, s in enumerate(SITES)}
    cols = parts["site119"][2]
    assert all(p[2] == cols for p in parts.values())
    res = {"feature_set": FS, "sites": list(SITES), "held_blocks": len(held), "loso": {}}
    for s in SITES:
        tr = [p for n, p in parts.items() if n != s]
        mdl = M.make("lgbm").fit(np.vstack([p[0] for p in tr]), np.concatenate([p[1] for p in tr]))
        pp = mdl.predict_proba(parts[s][0])[:, 1]
        res["loso"][s] = {k: round(float(v), 4) for k, v in M.metrics(parts[s][1], pp).items()}
        log(f"left out {s}: F1 {res['loso'][s]['f1']:.4f} P {res['loso'][s]['precision']:.3f} R {res['loso'][s]['recall']:.3f}")
    mdl = M.make("lgbm").fit(np.vstack([p[0] for p in parts.values()]), np.concatenate([p[1] for p in parts.values()]))
    joblib.dump({"model": mdl, "feature_set": FS, "columns": cols, "threshold": 0.5, "train_sites": list(SITES)}, OUT / "model.joblib")
    # held-out reviewed 2026 pixels: old (2025-only multi model) vs new
    d, Fr, c2 = review_features()
    assert c2 == cols
    d["p_new"] = np.round(100 * mdl.predict_proba(Fr)[:, 1]).astype(int)
    tab = d.groupby(["group", "answer"]).agg(n=("p", "size"), old_cane=("p", lambda v: int((v >= 50).sum())),
                                             new_cane=("p_new", lambda v: int((v >= 50).sum()))).reset_index()
    res["review_heldout"] = tab.to_dict("records")
    sure = d[d.answer != "unsure"]
    y = (sure.answer == "cane").astype(int)
    for k, col in (("old", "p"), ("new", "p_new")):
        yh = (sure[col] >= 50).astype(int)
        res[f"review_{k}"] = {"n": int(len(sure)), "acc": round(float((yh == y).mean()), 3),
                              "cane_recall": round(float(yh[y == 1].mean()), 3), "noncane_correct": round(float(1 - yh[y == 0].mean()), 3)}
    d.to_csv(OUT / "review_heldout_predictions.csv", index=False)
    (OUT / "cv.json").write_text(json.dumps(res, indent=1, default=str))
    print(tab.to_string(index=False)); print("old", res["review_old"], "\nnew", res["review_new"])
    log("saved", OUT / "model.joblib")


if __name__ == "__main__":
    main()

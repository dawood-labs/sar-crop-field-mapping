"""Standing-cane model trained on SEVERAL districts (2025 season), tested on a district it never saw.

Why: the model trained on the first AOI alone over-predicted cane by 37 % in a new district (Site21):
the non-cane there (cassava, upland crops, other rice systems) looks different. Training on more districts teaches
the model more kinds of non-cane.
Train sites (each with its own label QC, `label_qc.py`): first AOI (repo root) + sites listed in TRAIN_SITES.
Features: VV + VH 15-day composites Jan - 10 Oct 2025 (76 bands, same as the single-site model); LightGBM defaults.
Samples: per site 100k cane + 100k non-cane (clean label, edge >= 2).
Checks:
  cv_blocks  5 folds by 2 km block inside every site (all sites pooled)
  loso       leave-one-site-out: train on the other sites, predict the left-out site (transfer to a new place)
Final model on all train sites -> data/model/multi2025/model.joblib (same format as m2025, used by apply_site.py).
usage: python scripts/model_multi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os, time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

import model_2025 as M  # noqa: E402
import site_paths as SP

PROJ = M.PROJ
OUT = PROJ / "data/model/multi2025"
FS = os.environ.get("FEATURE_SET", "VVVH@202501a")
TRAIN_SITES = os.environ.get("TRAIN_SITES", "site21,site01").split(",")
N_PER_CLASS, SEED = 100_000, 0
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def site_paths(name):
    if name == "site99":
        return PROJ, PROJ / "data/features/s1_15d_2025.tif"
    root = SP.site_dir(name)
    return root, root / "data/features/s1_15d_2025jan.tif"


def site_samples(name, k):
    root, feat = site_paths(name)
    with rasterio.open(root / "data/labels/cane2025_qc.tif") as r:
        clean, edge = r.read(1), r.read(7)
    with rasterio.open(feat) as r:
        names, X = list(r.descriptions), r.read()
    H, W = clean.shape
    Xf = X.reshape(len(names), -1)
    ok = (Xf > -32000).all(0)
    rng = np.random.default_rng(SEED + k)
    idx = []
    for cls in (1, 0):
        c = np.flatnonzero((clean.ravel() == cls) & (edge.ravel() >= 2) & ok)
        idx.append(rng.choice(c, min(N_PER_CLASS, len(c)), replace=False))
    idx = np.sort(np.concatenate(idx))
    F, cols = M.build(Xf[:, idx].T, names, FS)
    rr, cc = np.divmod(idx, W)
    blk = (rr // M.CV_BLOCK) * ((W + M.CV_BLOCK - 1) // M.CV_BLOCK) + cc // M.CV_BLOCK
    fold = np.random.default_rng(SEED + 100 + k).integers(0, M.FOLDS, blk.max() + 1)[blk]
    log(f"{name}: {len(idx):,} px ({int(clean.ravel()[idx].sum()):,} cane), {len(cols)} features")
    return F, clean.ravel()[idx].astype(int), fold.astype("int8"), cols


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sites = ["site99"] + [s for s in TRAIN_SITES if s]
    parts = [site_samples(s, k) for k, s in enumerate(sites)]
    cols = parts[0][3]
    assert all(p[3] == cols for p in parts), "feature columns differ between sites"
    F = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    fold = np.concatenate([p[2] for p in parts]); site = np.concatenate([np.full(len(p[1]), k) for k, p in enumerate(parts)])
    res = {"feature_set": FS, "train_sites": sites, "n": int(len(y))}
    # pooled block CV
    oof, m = M.cv("lgbm", F, y, fold)
    res["cv_blocks_pooled"] = {k: round(v, 4) for k, v in m.items()}
    res["cv_blocks_per_site"] = {s: {k: round(v, 4) for k, v in M.metrics(y[site == k], oof[site == k]).items()} for k, s in enumerate(sites)}
    log("block CV pooled", res["cv_blocks_pooled"])
    # leave one site out
    res["loso"] = {}
    for k, s in enumerate(sites):
        mdl = M.make("lgbm").fit(F[site != k], y[site != k])
        p = mdl.predict_proba(F[site == k])[:, 1]
        res["loso"][s] = {kk: round(v, 4) for kk, v in M.metrics(y[site == k], p).items()}
        log(f"left out {s}: F1 {res['loso'][s]['f1']:.4f} P {res['loso'][s]['precision']:.3f} R {res['loso'][s]['recall']:.3f}")
    # final
    mdl = M.make("lgbm").fit(F, y)
    joblib.dump({"model": mdl, "feature_set": FS, "columns": cols, "threshold": 0.5, "train_sites": sites}, OUT / "model.joblib")
    (OUT / "cv.json").write_text(json.dumps(res, indent=1))
    log("saved", OUT / "model.joblib")


if __name__ == "__main__":
    main()

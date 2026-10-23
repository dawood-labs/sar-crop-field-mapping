"""Standing cane 2026: apply the final multi-district model to the 2026 season of the first AOI and compare with 2025.

The model was trained on 2025 half-months 2025-01a .. 2025-10a; the 2026 composites cover the same half-months one
year later, so band names are shifted by one year (2026MMx -> 2025MMx) before the column check.
Inputs : sites/site37_2026/data/features/s1_15d_2026jan.tif, data/features/s1_15d_2025.tif (2025, same grid),
         data/model/multi2025/model.joblib (MODEL to override), data/labels/cane2025_labels.tif (2025 map, AOI mask)
Outputs: sites/site37_2026/data/model/<tag>/
           cane_prob_2026.tif, cane_map_2026.tif     p x 100 and 1 / 0 (255 outside AOI) (+ .qml)
           cane_prob_2025.tif, cane_map_2025.tif     the same model on 2025 (like-for-like change, + .qml)
           change_2025_2026.tif                      0 none both, 1 cane both, 2 cane 2025 only (lost), 3 cane 2026 only (new)
           summary.json                              acres per year and change class; 2025 label acres
usage: [MODEL=...] [FEAT=nov] python scripts/apply_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

import model_2025 as M  # noqa: E402
import qgis_style as Q  # noqa: E402
import site_paths as SP

PROJ = M.PROJ
SITE = SP.site_dir("site37_2026")
MODEL = Path(os.environ.get("MODEL") or PROJ / "data/model/multi2025/model.joblib")
ACRE = 100 / 4046.8564224
FEAT = os.environ.get("FEAT", "jan")                        # "nov": composites s1_15d_2025nov / s1_15d_2026nov


def predict(feat, J, inside, shift_year=False, s2root=None, year=None):
    with rasterio.open(feat) as r:
        names, X, prof = list(r.descriptions), r.read(), r.profile.copy()
    if shift_year:
        names = [n[:-7] + str(int(n[-7:-3]) - 1) + n[-3:] for n in names]     # ..._2026MMx -> ..._2025MMx
    H, W = inside.shape
    Xf = X.reshape(len(names), -1)
    idx = np.flatnonzero(inside.ravel())
    n_missing = int((Xf[:, idx] <= -32000).any(0).sum())          # px with a half-month missing on one track
    F, cols = M.build(Xf[:, idx].T, names, J["feature_set"])
    if J.get("s2feat"):                                            # model trained with dry-season S2 features
        import model_loop as ML
        S2 = ML.s2_block(s2root, year)
        F = np.hstack([F, S2[0][:, idx].T]); cols = cols + S2[1]
    F[F <= -320] = np.nan                                          # LightGBM routes missing values natively
    assert cols == J["columns"], "feature columns differ from the trained model"
    p = np.full(H * W, 255, "uint8")
    p[idx] = np.round(100 * J["model"].predict_proba(F)[:, 1]).astype("uint8")
    p = p.reshape(H, W)
    cane = np.where(p == 255, 255, (p >= round(100 * J["threshold"])).astype("uint8")).astype("uint8")
    return p, cane, prof, n_missing


def main():
    J = joblib.load(MODEL)
    tag = MODEL.parent.name
    out = SITE / "data/model" / tag; out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(PROJ / "data/labels/cane2025_labels.tif") as r:
        lab = r.read(1)
    inside = lab != 255
    f26, f25 = (("s1_15d_2026nov.tif", "s1_15d_2025nov.tif") if FEAT == "nov" else ("s1_15d_2026jan.tif", "s1_15d_2025.tif"))
    p26, c26, prof, nd26 = predict(SITE / "data/features" / f26, J, inside, shift_year=True, s2root=SITE, year=2026)
    p25, c25, _, nd25 = predict(PROJ / "data/features" / f25, J, inside, s2root=PROJ, year=2025)
    chg = np.full(lab.shape, 255, "uint8")
    ok = inside & (c25 != 255) & (c26 != 255)
    for v, (a, b) in enumerate(((0, 0), (1, 1), (1, 0), (0, 1))):
        chg[ok & (c25 == a) & (c26 == b)] = v
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    for name, a in (("cane_prob_2026", p26), ("cane_map_2026", c26), ("cane_prob_2025", p25), ("cane_map_2025", c25),
                    ("change_2025_2026", chg)):
        with rasterio.open(out / f"{name}.tif", "w", **pr) as dst:
            dst.write(a, 1)
    for y in ("2025", "2026"):
        Q.pseudocolor(out / f"cane_prob_{y}.tif", Q.PROB_STOPS)
        Q.paletted(out / f"cane_map_{y}.tif", Q.MAP_CLASSES)
    Q.paletted(out / "change_2025_2026.tif", {0: ("#e6e6e6", "non-cane both years"), 1: ("#33ef2a", "cane both years"),
                                               2: ("#e63333", "cane 2025 only (lost in 2026)"), 3: ("#1a66ff", "cane 2026 only (new)")})
    a25, a26 = float((c25 == 1).sum() * ACRE), float((c26 == 1).sum() * ACRE)
    s = {"model": str(MODEL), "train_sites": J.get("train_sites"), "threshold": J["threshold"],
         "aoi_acres": round(float(inside.sum() * ACRE)), "px_with_a_missing_half_month": {"2025": nd25, "2026": nd26},
         "label_2025_cane_acres": round(float((lab == 1).sum() * ACRE)),
         "cane_acres": {"2025_model": round(a25), "2026_model": round(a26)}, "change_pct": round(100 * (a26 - a25) / a25, 1),
         "change_acres": {k: round(float((chg == v).sum() * ACRE)) for k, v in
                          (("noncane_both", 0), ("cane_both", 1), ("lost_2025_only", 2), ("new_2026_only", 3))},
         "uncertain_px_share_2026": round(float(((p26 >= 40) & (p26 < 60) & inside).sum() / inside.sum()), 4)}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()

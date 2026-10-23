"""External test: apply the final 2025 model (trained on the first AOI only) to another district and score it.

Inputs : <site>/data/features/s1_15d_2025jan.tif (same 15-day composites, Jan - 10 Oct 2025, same two tracks),
         <site>/data/labels/cane2025_labels.tif (raw 2025 cane map: label, poly_id, edge_px),
         data/model/m2025/model.joblib (final model of the first AOI: never saw this district)
Outputs (<site>/data/model/):
  cane_prob_2025.tif, cane_map_2025.tif, map_vs_label_2025.tif (+ .qml styles)
  external_test.json   P / R / F1 / OA / kappa vs the RAW label (all px and interior edge >= 2), area, 500 m blocks
usage: [MODEL=data/model/multi2025/model.joblib OUT_TAG=multi2025] python scripts/apply_site.py sites/<name>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

import model_2025 as M  # noqa: E402
import qgis_style as Q  # noqa: E402
import site_paths as SP


def main():
    site = SP.site_dir(sys.argv[1])
    model_path = Path(os.environ.get("MODEL") or M.OUT / "model.joblib")   # MODEL = e.g. data/model/multi2025/model.joblib
    J = joblib.load(model_path)
    J["model"].set_params(n_jobs=safe_resources.cpus())                      # predict with every usable core (result independent of thread count)
    feat = os.environ.get("FEATURES_FILE", "s1_15d_2025jan.tif")          # 2026: s1_15d_2026jan.tif
    yr = os.environ.get("YEAR", "2025")
    with rasterio.open(site / "data/features" / feat) as r:
        names, X, prof = list(r.descriptions), r.read(), r.profile.copy()
    if yr != "2025":                                                       # model columns are 2025 half-months
        names = [n[:-7] + str(int(n[-7:-3]) - (int(yr) - 2025)) + n[-3:] for n in names]
    labf = site / "data/labels/cane2025_labels.tif"
    if labf.exists():
        with rasterio.open(labf) as r:
            lab, edge = r.read(1).astype("uint8"), r.read(3)
    else:                                                                  # analysis-only district without a 2025 label: AOI mask only
        with rasterio.open(site / "data/grid/index.tif") as r:
            lab = np.where(r.read(2) > 0, 0, 255).astype("uint8"); edge = np.zeros(lab.shape, "uint8")
    H, W = lab.shape
    Xf = X.reshape(len(names), -1)
    inside = (lab != 255).ravel()
    idx = np.flatnonzero(inside & (Xf > -32000).all(0))
    F, cols = M.build(Xf[:, idx].T, names, J["feature_set"])
    if J.get("s2feat"):                                                    # model trained with dry-season S2 features
        import model_loop as ML
        S2 = ML.s2_block(site, int(yr))
        F = np.hstack([F, S2[0][:, idx].T]); cols = cols + S2[1]
    if J.get("pheno"):                                                     # round 3: date-free phenology features
        import model_loop as ML
        PH = ML.pheno_block(site, int(yr))
        F = np.hstack([F, PH[0][:, idx].T]); cols = cols + PH[1]
    F[F <= -320] = np.nan
    assert cols == J["columns"], "feature columns differ from the trained model"
    p = np.full(H * W, 255, "uint8")
    p[idx] = np.round(100 * J["model"].predict_proba(F)[:, 1]).astype("uint8")
    p = p.reshape(H, W)
    thr = round(100 * J["threshold"])
    cane = np.where(lab != 255, np.where(p == 255, 255, (p >= thr).astype("uint8")), 255).astype("uint8")
    cmp_ = np.full((H, W), 255, "uint8")
    ok = (lab != 255) & (cane != 255)
    for v, (a, b) in enumerate(((0, 0), (1, 1), (1, 0), (0, 1))):
        cmp_[ok & (lab == a) & (cane == b)] = v
    out = site / "data/model" / os.environ.get("OUT_TAG", ""); out.mkdir(parents=True, exist_ok=True)
    pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    for name, a in ((f"cane_prob_{yr}", p), (f"cane_map_{yr}", cane), (f"map_vs_label_{yr}", cmp_)):
        with rasterio.open(out / f"{name}.tif", "w", **pr) as dst:
            dst.write(a, 1)
    Q.pseudocolor(out / f"cane_prob_{yr}.tif", Q.PROB_STOPS)
    Q.paletted(out / f"cane_map_{yr}.tif", Q.MAP_CLASSES)
    Q.paletted(out / f"map_vs_label_{yr}.tif", Q.COMPARE_CLASSES)
    y, pp, e = lab[ok], p[ok] / 100, edge[ok]
    res = {"site": site.name, "model": str(model_path), "train_sites": J.get("train_sites", ["site37"]), "feature_set": J["feature_set"], "threshold": J["threshold"],
           "px_scored": int(ok.sum()), "px_no_data": int(((lab != 255) & (cane == 255)).sum())}
    for scope, s in (("raw_all", np.ones(len(y), bool)), ("raw_interior_edge>=2", e >= 2)):
        if labf.exists() and s.sum() > 0 and len(np.unique(y[s])) == 2:
            res[scope] = {k: round(float(v), 4) for k, v in M.metrics(y[s], pp[s], J["threshold"]).items()} | {"n": int(s.sum())}
        else:
            res[scope] = {"n": int(s.sum()), "note": "no 2025 label for this district: map only"}
    ac = 100 / 4046.856
    res["area_acres"] = {"label_raw": round(float((lab == 1).sum() * ac)), "predicted": round(float((cane == 1).sum() * ac)),
                         "aoi": round(float((lab != 255).sum() * ac))}
    rr, cc = np.nonzero(ok)
    b = (rr // 50) * ((W + 49) // 50) + cc // 50
    g = pd.DataFrame({"b": b, "y": y, "yh": (pp >= J["threshold"]).astype(int)}).groupby("b").agg(n=("y", "size"), lab=("y", "mean"), pred=("yh", "mean"))
    g = g[g.n >= 500]
    res["blocks_500m"] = {"n": int(len(g)), "r": round(float(np.corrcoef(g.lab, g.pred)[0, 1]), 4) if labf.exists() and len(g) > 1 else None,
                          "mae_cane_share": round(float((g.lab - g.pred).abs().mean()), 4),
                          "bias_pred_minus_label": round(float((g.pred - g.lab).mean()), 4)}
    qc = site / "data/labels/cane2025_qc.tif"          # labels QC-ed on THIS year's data (2026: 2025 map checked on 2026)
    if qc.exists():
        clean = rasterio.open(qc).read(1)[ok]
        s = clean != 255
        res["vs_clean_label"] = {k: round(float(v), 4) for k, v in M.metrics(clean[s], pp[s], J["threshold"]).items()} | {"n": int(s.sum())}
    res["year"] = yr
    (out / f"external_test_{yr}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

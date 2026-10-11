"""Cane-Cassava (sugarcane + cassava / upland field crops, one class: user 2026-10-09) vs the rest, SAR-only (S1 + terrain; no S2 anywhere in the model or in picking samples).

usage: cane_model.py cane_v1 | cane_v2 | cane_v3
Features: s1_features_v4.tif (84) + s1_features_v6_extra.tif (6: VH Jan->Mar rise, late cut, dry amplitude) + elevation, slope.
Labels (one per pixel, priority in this order):
  user check of rice v5 (1348570, 988937 = sugarcane) > GCP (3x3; Sugarcane or Cassava = 1) > blind cane reviews of earlier versions
  (data/model/cane_v*/review/review_results.csv, high/medium) > rice audits v1-v4 (high/medium; sugarcane / upland_crop = 1,
  every other subtype except mixed = 0) > labels_v1 candidates (high/medium; Sugarcane / Cassava = 1, others = 0)
  > WorldCover non-crop checked with SAR (tree / built / dark built / water, as rice v5) = 0.
XGBoost, 5-fold StratifiedGroupKFold (500 m blocks; GCP by point). Map = probability >= THR (0.40) and rice v5 probability < RICE_MAX (70) (see THR below).
Outputs data/model/<ver>/: cane_prob.tif, cane_map.tif, model.json, feature_importance.csv, samples.parquet,
  cv.json (out-of-fold by source, overlap with rice v5); then run field_map_v5.py <ver> Cane-Cassava cane_map.tif.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
import safe_resources  # noqa: E402,F401
from rice_model import worldcover, spaced_sample  # noqa: E402

VER = sys.argv[1] if len(sys.argv) > 1 else "cane_v1"
OUT = PROJ / "data/model" / VER
K = int(VER.split("_v")[1])
SEED, BLOCK, W = 0, 50, 2219
ACRE = 100 / 4046.856
USER_CANE = [1348570, 988937]
# map rule (user 2026-10-09: cane and cassava may mix, nothing else may): prob >= THR and rice v5 prob < RICE_MAX.
# Tested on review 1 (v2 out-of-fold): 0.50 -> P 0.81 R 0.60; 0.40 -> P 0.81 R 0.66; 0.40 + rice guard 70 -> P 0.86 R 0.57,
# and on all older labels P 0.72 -> 0.89 (false cane on rice fields removed).
# cane_v3: on review 2 (v3 OOF) 0.50 beats 0.40 (P 0.874 vs 0.821, R 0.715 vs 0.752, acc 0.917 vs 0.912) and beats
# v2 as published (P 0.856, R 0.706) -> 0.50 from v3 on.
THR, RICE_MAX = {1: 0.50, 2: 0.40}.get(K, 0.50), 70


def rule(p, rice_p):
    return (p >= THR) & (rice_p < RICE_MAX)


def metrics(y, p):
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[1, 0], zero_division=0)
    return {"n": int(len(y)), "n_cane": int(y.sum()), "acc": float((y == p).mean()), "cane_P": pr[0], "cane_R": rc[0], "cane_F1": f[0]}


def labels():
    S = [dict(row=p // W, col=p % W, y=1, source="user", cls="sugarcane", sid=f"user_{p}") for p in USER_CANE]
    g = pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv"); g = g[g.usable_for_training]
    S += [dict(row=r.row + dr, col=r.col + dc, y=int(r.class_2026 in ("Sugarcane", "Cassava")), source="gcp", cls=r.class_2026.lower(),
               sid=r.gcp_id, group=r.gcp_id) for r in g.itertuples() for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    for k in range(K - 1, 0, -1):                                    # newest review first
        f = PROJ / f"data/model/cane_v{k}/review/review_results.csv"
        if f.exists():
            a = pd.read_csv(f); a = a[a.label.isin(["Cane-Cassava", "Not-cane-cassava"]) & a.confidence.isin(["high", "medium"])]
            S += [dict(row=r.row, col=r.col, y=int(r.label == "Cane-Cassava"), source=f"review_v{k}", cls=r.subtype,
                       sid=f"cane_v{k}_{r.review_id}") for r in a.itertuples()]
    for v in (4, 3, 2, 1):
        a = pd.read_csv(PROJ / f"data/model/rice_v{v}/audit/audit_results.csv")
        a = a[a.label.isin(["Rice", "Non-rice"]) & a.confidence.isin(["high", "medium"]) & (a.subtype != "mixed")]
        S += [dict(row=r.row, col=r.col, y=int(r.subtype in ("sugarcane", "upland_crop")), source=f"audit_v{v}", cls=r.subtype,
                   sid=f"rice_v{v}_{r.audit_id}") for r in a.itertuples()]
    key = pd.read_csv(PROJ / "data/labels_v1/key/candidates_key.csv")
    c = pd.concat([pd.read_csv(f) for f in sorted((PROJ / "data/labels_v1/labels").glob("batch_*.csv"))]).merge(key, on="audit_id")
    c = c[c.confidence.isin(["high", "medium"]) & (c.label != "Unsure")]
    S += [dict(row=r.row, col=r.col, y=int(r.label in ("Sugarcane", "Cassava")), source="candidates", cls=r.label.lower(),
               sid="cand_" + r.audit_id) for r in c.itertuples()]
    return S


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    grid = json.loads((PROJ / "data/grid/grid.json").read_text())
    with rasterio.open(PROJ / "data/features/s1_features_v4.tif") as r:
        X = r.read(); names = list(r.descriptions); prof = r.profile.copy()
    for f in ("data/features/s1_features_v6_extra.tif", "data/ancillary/dem.tif"):
        with rasterio.open(PROJ / f) as r:
            X = np.concatenate([X, r.read().astype("float32")]); names += list(r.descriptions) if "v6" in f else ["elevation_m", "slope_deg"]
    nx = {n: i for i, n in enumerate(names)}
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    H, Wd = inside.shape
    print(f"features {X.shape} in {time.time() - t0:.0f} s", flush=True)

    S = labels()
    avoid = pd.DataFrame(S)[["row", "col"]].to_numpy()
    # WorldCover non-crop, checked with SAR (same rules as rice v4/v5)
    tm = lambda v, months: np.mean([X[nx[f"{t}_{v}_{m}"]] for t in ("RO164_DES", "RO099_ASC") for m in months], 0)  # noqa: E731
    wc = worldcover(grid)
    vh = np.stack([tm("VH", [f"2026{m:02d}"]) for m in range(3, 10)])
    vh_mean, vh_max, vh_std = vh.mean(0), vh.max(0), vh.std(0); del vh
    vv_mean = tm("VV", [f"2026{m:02d}" for m in range(3, 10)]); stable = vh_std <= 1
    rules = {"tree": ((wc == 10) & stable & (vh_mean >= -15.5), 200), "built": ((wc == 50) & stable & (vh_mean >= -14), 120),
             "built_dark": ((wc == 50) & stable & (vh_mean < -14) & (vv_mean >= -9), 80), "water": ((wc == 80) & (vh_max <= -21), 80)}
    for cls, (m, n) in rules.items():
        rr, cc = np.nonzero(m & inside)
        pts = spaced_sample(rr, cc, n, 30, rng, avoid) if len(rr) else np.empty((0, 2))
        S += [dict(row=int(p[0]), col=int(p[1]), y=0, source="worldcover", cls=cls, sid=f"wc_{cls}_{i}") for i, p in enumerate(pts)]
    del wc, vh_mean, vh_max, vh_std, vv_mean, stable
    S = pd.DataFrame(S)
    S = S[(S.row >= 0) & (S.row < H) & (S.col >= 0) & (S.col < Wd)]
    order = ["user", "gcp"] + [f"review_v{k}" for k in range(K - 1, 0, -1)] + ["audit_v4", "audit_v3", "audit_v2", "audit_v1", "candidates", "worldcover"]
    S = S.assign(o=S.source.map({s: i for i, s in enumerate(order)})).sort_values("o", kind="stable").drop_duplicates(["row", "col"]).drop(columns="o")
    S["group"] = np.where(S.source == "gcp", S.get("group"), "blk" + (S.row // BLOCK).astype(str) + "_" + (S.col // BLOCK).astype(str))
    S = S.reset_index(drop=True)
    F = X[:, S.row, S.col].T
    ok = np.isfinite(F).all(1); S, F = S[ok].reset_index(drop=True), F[ok]
    print(f"samples {len(S)} px (cane {int(S.y.sum())}); by source:", S.groupby(["source", "y"]).size().to_dict(), flush=True)

    from sklearn.model_selection import StratifiedGroupKFold
    from xgboost import XGBClassifier
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                  n_jobs=os.cpu_count(), random_state=SEED, eval_metric="logloss")
    spw = (1 - S.y).sum() / max(S.y.sum(), 1)
    oof = np.zeros(len(S))
    for trn, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, S.y, S.group):
        oof[te] = XGBClassifier(**params, scale_pos_weight=spw).fit(F[trn], S.y.iloc[trn]).predict_proba(F[te])[:, 1]
    S["oof_prob"] = oof
    rice_prob = rasterio.open(PROJ / "data/model/rice_v5/rice_prob.tif").read(1)
    S["rice_p"] = rice_prob[S.row, S.col]
    S["oof_pred"] = rule(S.oof_prob, S.rice_p).astype(int)
    nw = S[S.source != "worldcover"]
    cv = {"rule": f"prob >= {THR} and rice_v5 prob < {RICE_MAX}",
          "oof_all_labels_except_worldcover": metrics(nw.y.to_numpy(), nw.oof_pred.to_numpy()),
          "oof_by_source": {s: metrics(q.y.to_numpy(), q.oof_pred.to_numpy()) for s, q in S.groupby("source")},
          "oof_false_cane_by_class": nw[nw.y == 0].groupby("cls").oof_pred.mean().round(3).to_dict()}
    if K > 1:      # honest test on the latest review: previous published map (never saw it) vs this model out-of-fold
        P = PROJ / f"data/model/cane_v{K - 1}"
        key = pd.read_csv(P / "review/key/review_key.csv")[["review_id", "bin"]].assign(sid=lambda q: f"cane_v{K - 1}_" + q.review_id)
        T = S[S.source == f"review_v{K - 1}"].merge(key, on="sid")
        T["prev_pred"] = (rasterio.open(P / "cane_map.tif").read(1)[T.row, T.col] == 1).astype(int)   # as published
        T["new_pred"] = T.oof_pred
        wts = {r["bin"]: r["map_acres"] for r in json.loads((P / "review/review_summary.json").read_text())["per_bin_area"]}
        aw = {}
        for m in ("prev_pred", "new_pred"):
            cm = np.zeros((2, 2))
            for b, q in T.groupby("bin"):
                for yy in (0, 1):
                    for pp in (0, 1):
                        cm[yy, pp] += wts.get(b, 0) / len(q) * ((q.y == yy) & (q[m] == pp)).sum()
            aw[m] = {"acc": cm.trace() / cm.sum(), "cane_P": cm[1, 1] / max(cm[:, 1].sum(), 1e-9), "cane_R": cm[1, 1] / max(cm[1].sum(), 1e-9)}
        cv["honest_test_latest_review"] = {"n": int(len(T)), f"cane_v{K - 1}_published": metrics(T.y.to_numpy(), T.prev_pred.to_numpy()),
                                           f"{VER}_oof": metrics(T.y.to_numpy(), T.new_pred.to_numpy()),
                                           "area_weighted": {f"cane_v{K - 1}_published": aw["prev_pred"], f"{VER}_oof": aw["new_pred"]}}
    print(json.dumps({k: cv[k] for k in ("oof_all_labels_except_worldcover", "oof_false_cane_by_class", "honest_test_latest_review") if k in cv},
                     indent=1, default=lambda v: round(float(v), 3)), flush=True)

    model = XGBClassifier(**params, scale_pos_weight=spw).fit(F, S.y)
    model.save_model(OUT / "model.json")
    pd.Series(model.feature_importances_, index=names).sort_values(ascending=False).to_csv(OUT / "feature_importance.csv", header=["importance"])
    flat = X.reshape(len(names), -1)
    valid = inside.ravel().copy()
    for b in range(len(names)):
        valid &= np.isfinite(flat[b])
    prob = np.full(H * Wd, 255, np.uint8); idx = np.nonzero(valid)[0]
    for s in range(0, len(idx), 500_000):
        b = idx[s:s + 500_000]
        prob[b] = np.round(model.predict_proba(flat[:, b].T)[:, 1] * 100).astype(np.uint8)
    del X, flat
    prob = prob.reshape(H, Wd)
    cane = np.where(prob == 255, 255, rule(prob / 100, rice_prob).astype(np.uint8)).astype(np.uint8)
    p1 = dict(prof, count=1, dtype="uint8", nodata=255, predictor=2)
    for name, arr, desc in [("cane_prob.tif", prob, "cane-cassava probability x100"), ("cane_map.tif", cane, f"1 cane-cassava (prob >= {THR}, rice v5 prob < {RICE_MAX}), 0 not")]:
        with rasterio.open(OUT / name, "w", **p1) as dst:
            dst.write(arr, 1); dst.set_band_description(1, desc)
    rice = rasterio.open(PROJ / "data/model/rice_v5/rice_map.tif").read(1)   # (rice_prob read above)
    cv["area"] = {"cane_acres": float((cane == 1).sum() * ACRE), "cane_and_rice_v5_acres": float(((cane == 1) & (rice == 1)).sum() * ACRE),
                  "rice_v5_acres": float((rice == 1).sum() * ACRE)}
    if K > 1:
        prev = rasterio.open(PROJ / f"data/model/cane_v{K - 1}/cane_map.tif").read(1)
        cv["area"].update(prev_cane_acres=float((prev == 1).sum() * ACRE), added_acres=float(((cane == 1) & (prev == 0)).sum() * ACRE),
                          removed_acres=float(((cane == 0) & (prev == 1)).sum() * ACRE))
    (OUT / "cv.json").write_text(json.dumps(cv, indent=1, default=float))
    S.to_parquet(OUT / "samples.parquet")
    print("area:", {k: round(v) for k, v in cv["area"].items()}, f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

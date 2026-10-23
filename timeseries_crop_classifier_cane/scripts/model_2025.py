"""Standing-cane model on the 2025 season: feature-set comparison, model comparison, tuning, held-out block test.

Inputs : data/features/s1_15d_2025.tif (162 bands: 27 half-months x VV / VH / VH-VV x 2 tracks, dB x 100)
         data/labels/cane2025_qc.tif   (label_clean 1 / 0 / 255, block_id 500 m, edge_px) + cane2025_labels.tif (raw)
Split  : 2 km blocks; 20 % of them = TEST (never used for any choice); the rest = TRAIN, 5 group folds by 2 km block.
Samples: TRAIN 300k px (150k cane / 150k non-cane, clean, edge >= 2); TEST = every labelled px of the test blocks.
Stages (each cached in data/model/m2025/, rerun with --force <stage>):
  sets    LightGBM, 5-fold CV, 8 feature sets: VV, VH, VVVH, VHmVV, each without / with temporal stats
  models  best set: logistic / RandomForest / XGBoost / LightGBM, same folds
  tune    Optuna on the best boosted model (folds 0-2, 150k px), then threshold from out-of-fold p (max F1)
  test    final model on all TRAIN px -> TEST blocks: P / R / F1 / OA / kappa, area, 500 m block agreement;
          against the CLEAN labels (optimistic: QC used model disagreement) and the RAW labels (pessimistic)
  map     p(cane) for the whole AOI -> data/model/m2025/cane_prob_2025.tif (uint8 x 100) + cane_map_2025.tif
  window  start-month sweep (Sep 2024 .. Jun 2025, end fixed 10 Oct 2025) with the best set; latest start within
          0.002 F1 of the best is kept
usage: python scripts/model_2025.py [sets|models|window|tune|test|map|all] [--force]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os, time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
OUT = PROJ / "data/model/m2025"
FEAT = PROJ / "data/features/s1_15d_2025.tif"
QC = PROJ / "data/labels/cane2025_qc.tif"
RAW = PROJ / "data/labels/cane2025_labels.tif"
NCPU = os.cpu_count() or 8
SEED, CV_BLOCK, TEST_FRAC, N_TRAIN, FOLDS = 0, 200, 0.2, 300_000, 5
FORCE = "--force" in sys.argv
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


# ------------------------------------------------------------------ data
def load_all():
    """Whole feature stack in memory (int16, ~4.5 GB) read in one multi-threaded call, plus labels."""
    with rasterio.open(FEAT) as r:
        names = list(r.descriptions)
        X = r.read()                                         # (162, H, W) int16
        prof = r.profile.copy()
    with rasterio.open(QC) as r:
        clean, block, edge = r.read(1).astype("uint8"), r.read(3), r.read(7).astype("uint8")
    raw = rasterio.open(RAW).read(1).astype("uint8")
    return names, X, clean, raw, block, edge, prof


def split(shape):
    H, W = shape
    rr, cc = np.indices(shape, dtype="int32")
    cvb = (rr // CV_BLOCK) * ((W + CV_BLOCK - 1) // CV_BLOCK) + cc // CV_BLOCK
    rng = np.random.default_rng(SEED + 11)
    nb = cvb.max() + 1
    is_test_b = rng.random(nb) < TEST_FRAC
    fold_b = rng.integers(0, FOLDS, nb)
    return cvb, is_test_b[cvb], fold_b[cvb].astype("int8")


def samples():
    f = OUT / "samples.npz"
    if f.exists() and not FORCE:
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    names, X, clean, raw, block, edge, _ = load_all()
    log("features in memory", X.shape)
    cvb, test, fold = split(clean.shape)
    rng = np.random.default_rng(SEED)
    Xf = X.reshape(len(names), -1)
    ok = (Xf > -32000).all(0)
    tr_idx = []
    for cls in (1, 0):
        c = np.flatnonzero((clean.ravel() == cls) & (edge.ravel() >= 2) & ~test.ravel() & ok)
        tr_idx.append(rng.choice(c, N_TRAIN // 2, replace=False))
    tr_idx = np.sort(np.concatenate(tr_idx))
    te_idx = np.flatnonzero(test.ravel() & (raw.ravel() != 255) & ok)
    d = {"names": np.array(names), "Xtr": Xf[:, tr_idx].T.copy(), "ytr": clean.ravel()[tr_idx], "ftr": fold.ravel()[tr_idx],
         "Xte": Xf[:, te_idx].T.copy(), "yte_clean": clean.ravel()[te_idx], "yte_raw": raw.ravel()[te_idx],
         "edge_te": edge.ravel()[te_idx], "block_te": block.ravel()[te_idx], "idx_te": te_idx,
         "n_test_blocks_2km": int(len(np.unique(cvb[test & (raw != 255)]))), "n_train_blocks_2km": int(len(np.unique(cvb[~test & (raw != 255)])))}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    log(f"samples: train {len(tr_idx):,} px, test {len(te_idx):,} px ({d['n_test_blocks_2km']} test / {d['n_train_blocks_2km']} train 2 km blocks)")
    return d


# ------------------------------------------------------------------ features
def parse(names):
    t = [n.split("_") for n in names]
    return pd.DataFrame({"track": [f"{a[0]}_{a[1]}" for a in t], "var": [a[2] for a in t], "period": [a[3] for a in t]})


def temporal_stats(X, names, var):
    """Per track: mean, std, min, max, p10, p90, argmin (period index), Dec-Apr min, Jun-Oct mean, dip = Jun-Oct mean -
    Dec-Apr min, Sep-Nov'24 mean, regrowth = Apr-May mean - Dec-Apr min."""
    p = parse(names)
    out, cols = [], []
    for t in sorted(p.track.unique()):
        sel = p[(p.track == t) & (p["var"] == var)]
        A = X[:, sel.index.values].astype("float32") / 100
        per = sel.period.values
        m = lambda keys: np.isin([q[:6] for q in per], keys)  # noqa: E731
        dry = m(["202412", "202501", "202502", "202503", "202504"])
        wet = m(["202506", "202507", "202508", "202509", "202510"])
        early = m(["202409", "202410", "202411"])
        reg = m(["202504", "202505"])
        dmin = A[:, dry].min(1)
        st = [A.mean(1), A.std(1), A.min(1), A.max(1), np.percentile(A, 10, 1), np.percentile(A, 90, 1),
              A.argmin(1).astype("float32"), dmin, A[:, wet].mean(1), A[:, wet].mean(1) - dmin, A[:, early].mean(1),
              A[:, reg].mean(1) - dmin]
        out += st
        cols += [f"{t}_{var}_{s}" for s in ("mean", "std", "min", "max", "p10", "p90", "argmin", "drymin", "wetmean", "dip", "early", "regrow")]
    return np.stack(out, 1), cols


FEATURE_SETS = ["VV", "VH", "VVVH", "VHmVV", "VV+stats", "VH+stats", "VVVH+stats", "VHmVV+stats"]


def build(X, names, fs):
    """fs = '<vars>[+stats][@<first period>]', e.g. 'VVVH@202412a' = VV + VH from Dec 2024 to the end."""
    fs, _, start = fs.partition("@")
    p = parse(names)
    base = fs.split("+")[0]
    vars_ = {"VV": ["VV"], "VH": ["VH"], "VVVH": ["VV", "VH"], "VHmVV": ["VHmVV"]}[base]
    keep = p["var"].isin(vars_) & ((p.period >= start) if start else True)
    idx = p.index[keep].values
    F, cols = X[:, idx].astype("float32") / 100, [names[i] for i in idx]
    if fs.endswith("+stats"):
        for v in vars_:
            S, c = temporal_stats(X, names, v)
            F = np.hstack([F, S]); cols += c
    return F, cols


# ------------------------------------------------------------------ models
def metrics(y, p, thr=0.5):
    from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score
    yh = (p >= thr).astype(int)
    return {"auc": roc_auc_score(y, p), "f1": f1_score(y, yh), "precision": precision_score(y, yh), "recall": recall_score(y, yh),
            "oa": float((yh == y).mean()), "kappa": cohen_kappa_score(y, yh), "area_ratio_pred_over_label": float(yh.sum() / max(y.sum(), 1))}


def make(kind, params=None):
    params = params or {}
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**{"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                                     "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5, "n_jobs": NCPU,
                                     "verbose": -1, "random_state": SEED, **params})
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(**{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8,
                                    "colsample_bytree": 0.5, "tree_method": "hist", "n_jobs": NCPU, "random_state": SEED,
                                    "eval_metric": "logloss", **params})
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**{"n_estimators": 300, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": NCPU,
                                         "random_state": SEED, **params})
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=params.get("C", 1.0)))
    raise ValueError(kind)


def cv(kind, F, y, folds, params=None, use_folds=None):
    oof = np.full(len(y), np.nan, "float32")
    for k in (use_folds if use_folds is not None else range(FOLDS)):
        tr, te = folds != k, folds == k
        m = make(kind, params).fit(F[tr], y[tr])
        oof[te] = m.predict_proba(F[te])[:, 1]
    ok = ~np.isnan(oof)
    return oof, metrics(y[ok], oof[ok])


def stage_sets(d):
    f = OUT / "sets.json"
    if f.exists() and not FORCE:
        return json.loads(f.read_text())
    names = list(d["names"]); res = {}
    for fs in FEATURE_SETS:
        F, cols = build(d["Xtr"], names, fs)
        _, m = cv("lgbm", F, d["ytr"], d["ftr"])
        res[fs] = {"n_features": len(cols), **{k: round(v, 4) for k, v in m.items()}}
        log(f"set {fs:12s} n={len(cols):3d}  AUC {m['auc']:.4f}  F1 {m['f1']:.4f}  P {m['precision']:.3f}  R {m['recall']:.3f}")
    f.write_text(json.dumps(res, indent=1))
    return res


def best_set(res):
    # best F1; prefer the smaller set when within 0.002
    order = sorted(res, key=lambda k: (-res[k]["f1"], res[k]["n_features"]))
    top = res[order[0]]["f1"]
    return min([k for k in order if res[k]["f1"] >= top - 0.002], key=lambda k: res[k]["n_features"])


def stage_models(d, fs):
    f = OUT / "models.json"
    if f.exists() and not FORCE:
        return json.loads(f.read_text())
    F, _ = build(d["Xtr"], list(d["names"]), fs)
    res = {}
    for kind in ("logreg", "rf", "xgb", "lgbm"):
        t = time.time()
        _, m = cv(kind, F, d["ytr"], d["ftr"])
        res[kind] = {**{k: round(v, 4) for k, v in m.items()}, "seconds": round(time.time() - t)}
        log(f"model {kind:7s} AUC {m['auc']:.4f}  F1 {m['f1']:.4f}  P {m['precision']:.3f}  R {m['recall']:.3f}  ({res[kind]['seconds']} s)")
    f.write_text(json.dumps({"feature_set": fs, "results": res}, indent=1))
    return {"feature_set": fs, "results": res}


def stage_window(d, fs):
    """Start month sweep (end fixed at the last half-month): does the season really need data from Sep of the
    previous year? Same folds, LightGBM default params."""
    f = OUT / "window.json"
    if f.exists() and not FORCE:
        return json.loads(f.read_text())
    periods = sorted(parse(list(d["names"])).period.unique())
    starts = [q for q in periods if q.endswith("a") and q <= "202506a"]
    res = {}
    for st in starts:
        F, cols = build(d["Xtr"], list(d["names"]), f"{fs}@{st}")
        _, m = cv("lgbm", F, d["ytr"], d["ftr"])
        res[st] = {"n_features": len(cols), **{k: round(v, 4) for k, v in m.items()}}
        log(f"window from {st}: n={len(cols):3d}  AUC {m['auc']:.4f}  F1 {m['f1']:.4f}  P {m['precision']:.3f}  R {m['recall']:.3f}")
    f.write_text(json.dumps(res, indent=1))
    return res


def best_window(res):
    """Latest start whose F1 is within 0.002 of the best (shorter series = less data to download and process)."""
    top = max(v["f1"] for v in res.values())
    return max(k for k, v in res.items() if v["f1"] >= top - 0.002)


def stage_tune(d, fs, kind, n_trials=30):
    f = OUT / "tune.json"
    if f.exists() and not FORCE:
        return json.loads(f.read_text())
    if os.environ.get("NO_TUNE") == "1":
        # user 12 Oct 2026: 12 Optuna trials spanned F1 0.960-0.962 (= fold noise) -> default LightGBM params
        oof, m = cv(kind, *build(d["Xtr"], list(d["names"]), fs)[:1], d["ytr"], d["ftr"])
        return threshold_result(kind, fs, {}, None, oof, m, d["ytr"], f)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    F, _ = build(d["Xtr"], list(d["names"]), fs)
    y, folds = d["ytr"], d["ftr"]
    rng = np.random.default_rng(SEED + 3)
    sub = rng.choice(len(y), 150_000, replace=False)
    Fs, ys, fds = F[sub], y[sub], folds[sub]

    def objective(trial):
        if kind == "lgbm":
            p = {"num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                 "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
                 "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
                 "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
        else:
            p = {"max_depth": trial.suggest_int("max_depth", 4, 12),
                 "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                 "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.9),
                 "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                 "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
        _, m = cv(kind, Fs, ys, fds, p, use_folds=[0, 1, 2])
        return m["f1"]

    # stored in SQLite so a pod restart resumes instead of starting over; every trial is logged
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                             storage=f"sqlite:///{OUT / 'tune_optuna.db'}", study_name=f"{kind}_{fs}", load_if_exists=True)
    left = n_trials - len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])
    cb = lambda s_, t: log(f"  trial {t.number}: F1 {t.value:.4f} (best {s_.best_value:.4f})")  # noqa: E731
    if left > 0:
        st.optimize(objective, n_trials=left, show_progress_bar=False, callbacks=[cb])
    log(f"tune {kind}: best F1 {st.best_value:.4f} {st.best_params}")
    # full 5-fold OOF with the best params -> threshold
    oof, m = cv(kind, F, y, folds, st.best_params)
    return threshold_result(kind, fs, st.best_params, st.best_value, oof, m, y, f)


def threshold_result(kind, fs, params, tune_f1, oof, m, y, f):
    """Threshold = max F1 on the 5-fold out-of-fold p (TRAIN only; the test blocks are never used)."""
    thr = np.linspace(0.2, 0.8, 61)
    f1s = [metrics(y, oof, t)["f1"] for t in thr]
    t_best = float(thr[int(np.argmax(f1s))])
    res = {"kind": kind, "feature_set": fs, "best_params": params, "tune_f1_3fold_150k": tune_f1,
           "oof_5fold_default_thr": {k: round(v, 4) for k, v in m.items()}, "threshold_maxF1": t_best,
           "oof_5fold_at_threshold": {k: round(v, 4) for k, v in metrics(y, oof, t_best).items()},
           "default_params_f1": None}
    f.write_text(json.dumps(res, indent=1))
    log(f"OOF F1 {m['f1']:.4f} at 0.5; threshold {t_best:.2f} -> F1 {res['oof_5fold_at_threshold']['f1']:.4f}")
    return res


def stage_test(d, tune):
    f = OUT / "test.json"
    if f.exists() and not FORCE:
        return json.loads(f.read_text())
    import joblib
    names = list(d["names"])
    F, cols = build(d["Xtr"], names, tune["feature_set"])
    m = make(tune["kind"], tune["best_params"]).fit(F, d["ytr"])
    joblib.dump({"model": m, "feature_set": tune["feature_set"], "columns": cols, "threshold": tune["threshold_maxF1"]}, OUT / "model.joblib")
    Fte, _ = build(d["Xte"], names, tune["feature_set"])
    p = m.predict_proba(Fte)[:, 1]
    np.save(OUT / "p_test.npy", p.astype("float32"))
    thr = tune["threshold_maxF1"]
    res = {"threshold": thr, "test_px": int(len(p))}
    for lab, y in (("clean", d["yte_clean"]), ("raw", d["yte_raw"])):
        for scope, s in (("all", np.ones(len(y), bool)), ("interior_edge>=2", d["edge_te"] >= 2)):
            s = s & (y != 255)
            res[f"{lab}_{scope}"] = {k: round(v, 4) for k, v in metrics(y[s], p[s], thr).items()} | {"n": int(s.sum())}
    # 500 m block agreement on raw labels: cane share per block, label vs prediction
    y, yh, b = d["yte_raw"], (p >= thr).astype(int), d["block_te"]
    g = pd.DataFrame({"b": b, "y": y, "yh": yh}).groupby("b").agg(n=("y", "size"), lab=("y", "mean"), pred=("yh", "mean"))
    g = g[g.n >= 500]
    res["blocks_500m"] = {"n": int(len(g)), "r": round(float(np.corrcoef(g.lab, g.pred)[0, 1]), 4),
                          "mae_cane_share": round(float((g.lab - g.pred).abs().mean()), 4),
                          "bias_pred_minus_label": round(float((g.pred - g.lab).mean()), 4)}
    res["area_test_acres"] = {"label_raw": round(float(y.sum() * 100 / 4046.856)), "predicted": round(float(yh.sum() * 100 / 4046.856))}
    f.write_text(json.dumps(res, indent=1))
    log("TEST", json.dumps(res, indent=1))
    return res


def stage_map(d):
    import joblib
    J = joblib.load(OUT / "model.joblib")
    names, X, clean, raw, block, edge, prof = load_all()
    H, W = clean.shape
    inside = raw != 255
    prob = np.full(H * W, 255, "uint8")
    Xf = X.reshape(len(names), -1)
    step = 400_000
    idx = np.flatnonzero(inside.ravel())
    for s in range(0, len(idx), step):
        ii = idx[s:s + step]
        F, _ = build(Xf[:, ii].T, names, J["feature_set"])
        prob[ii] = np.round(100 * J["model"].predict_proba(F)[:, 1]).astype("uint8")
    prob = prob.reshape(H, W)
    cane = np.where(inside, (prob >= round(100 * J["threshold"])).astype("uint8"), 255)
    p = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
    for name, a in (("cane_prob_2025", prob), ("cane_map_2025", cane)):
        with rasterio.open(OUT / f"{name}.tif", "w", **p) as dst:
            dst.write(a, 1)
    ac = 100 / 4046.856
    s = {"cane_acres_map": round(float((cane == 1).sum() * ac)), "cane_acres_label_2025": round(float((raw == 1).sum() * ac)),
         "aoi_acres": round(float(inside.sum() * ac)), "threshold": J["threshold"]}
    (OUT / "map_summary.json").write_text(json.dumps(s, indent=1))
    log("MAP", s)


def main():
    stage = next((a for a in sys.argv[1:] if not a.startswith("--")), "all")
    d = samples()
    sets = stage_sets(d)
    if stage == "sets":
        return
    fs = best_set(sets)
    log("best feature set:", fs)
    mods = stage_models(d, fs)
    if stage == "models":
        return
    win = stage_window(d, fs)
    st = os.environ.get("WINDOW_START") or best_window(win)   # user choice 12 Oct 2026: 202501a (Jan)
    log("start period:", st, "(fixed by WINDOW_START)" if os.environ.get("WINDOW_START") else "(best_window rule)")
    if stage == "window":
        return
    r = mods["results"]
    kind = "lgbm" if r["lgbm"]["f1"] >= r["xgb"]["f1"] else "xgb"
    tune = stage_tune(d, f"{fs}@{st}", kind)
    if stage == "tune":
        return
    stage_test(d, tune)
    if stage == "test":
        return
    stage_map(d)


if __name__ == "__main__":
    main()

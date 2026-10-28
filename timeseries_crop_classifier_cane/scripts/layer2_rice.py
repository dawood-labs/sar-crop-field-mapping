"""Layer 2, step 1: remove RICE from the cane-like map with the season (phenology) features.

Train: reviewed blocks (sites/layer2/points.parquet, split train): rice = 1, sugarcane / cassava / other_crop = 0, sea_* features
(= pheno_2026.tif bands). Leave-one-district-out CV; the test-set blocks are an extra holdout.
Apply: per district, inside the layer-1 mask (<src>/cane_map_2026.tif == 1): p_rice >= THR -> class 7 (rice), else 1 (cane-like).
Output <site>/data/model/<out>/: cane_map_2026.tif (1 cane-like after rice removal), class_2026.tif (1 sugarcane-like, 7 rice,
0 none; company codes) + .qml, rice_prob_2026.tif, layer2.json.  Model: sites/layer2/rice_model.joblib.
usage: python scripts/layer2_rice.py train            -> CV numbers, model, test-set effect
       python scripts/layer2_rice.py apply <src> <out> sites/<site> [...]
"""
import json, sys, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd, rasterio
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qgis_style as Q  # noqa: E402
import site_paths as SP
P = Path(__file__).resolve().parents[1]; L2 = P / "sites/layer2"; THR = float(__import__("os").environ.get("RICE_THR", "0.5"))
ACRE = 100 / 4046.8564224


def train():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
    df = pd.read_parquet(L2 / "points.parquet"); cols = [c for c in df.columns if c.startswith("sea_")]
    d = df[df.what.isin(["sugarcane", "cassava", "other_crop", "rice"])].copy(); d["y"] = (d.what == "rice").astype(int)
    tr, te = d[d.split == "train"], d[d.split == "test"]
    extra = __import__("os").environ.get("EXTRA_RICE")          # user polygons (analysis gpkg): rice calls = 1, cassava-like = 0
    if extra:
        import geopandas as gpd
        from rasterio.features import rasterize
        g = gpd.read_file(extra).to_crs(32648); SITE = {"site86": "site86_2026", "Site131": "site131_2026", "Site113": "site113_2026"}
        ex = []
        for sk, site in SITE.items():
            with rasterio.open(SP.site_dir(site) / "data/features/pheno_2026.tif") as r:
                names = list(r.descriptions); X = r.read().astype("float32"); trf = r.transform
            X[X <= -32000] = np.nan; uns = [i for i, n in enumerate(names) if n.split("_", 2)[-1] in ("last_cut", "grow_len", "grow_total") or n in ("greenup_days", "n_clear_mayoct")]
            X /= 100.0; X[uns] *= 100.0
            for _, q in g[g.site == sk].iterrows():
                if q.call2.startswith("unclear") or q.call2.startswith("bare"): continue
                mk = rasterize([(q.geometry, 1)], out_shape=X.shape[1:], transform=trf, dtype="uint8").astype(bool)
                px = X[:, mk].T
                e = pd.DataFrame(px, columns=["sea_" + n for n in names]); e["y"] = int(q.call2.startswith("rice")); e["site"] = site; ex.append(e)
        ex = pd.concat(ex, ignore_index=True); print(f"extra user-polygon px: {len(ex)} (rice {int(ex.y.sum())})")
        tr = pd.concat([tr, ex], ignore_index=True)
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=0)
    oof = np.zeros(len(tr))
    for site in tr.site.unique():
        m = (tr.site == site).values
        oof[m] = lgb.LGBMClassifier(**params).fit(tr.loc[~m, cols], tr.y[~m]).predict_proba(tr.loc[m, cols])[:, 1]
    p, r, f, _ = precision_recall_fscore_support(tr.y, oof >= THR, average="binary")
    print(f"LODO-cv rice: AUC {roc_auc_score(tr.y, oof):.3f}  P {p:.2f} R {r:.2f} F1 {f:.2f}  (n rice {int(tr.y.sum())} / {len(tr)})")
    mdl = lgb.LGBMClassifier(**params).fit(tr[cols], tr.y); joblib.dump({"model": mdl, "columns": cols, "thr": THR}, L2 / "rice_model.joblib")
    pt = mdl.predict_proba(te[cols])[:, 1]
    p, r, f, _ = precision_recall_fscore_support(te.y, pt >= THR, average="binary")
    print(f"testset rice: AUC {roc_auc_score(te.y, pt):.3f}  P {p:.2f} R {r:.2f} F1 {f:.2f}  (n rice {int(te.y.sum())} / {len(te)})")
    # effect on the layer-1 test-set score: cane-like prediction = rn5 map AND NOT rice
    key = pd.read_csv(P / "sites/testset_v1/score_s2on_r2_oct_rn5_rows.csv")
    tt = df[df.split == "test"].merge(key[["site", "row", "col", "label", "pred", "weight"]], on=["site", "row", "col"])
    tt = tt[tt.label >= 0]; tt["p_rice"] = mdl.predict_proba(tt[cols])[:, 1]
    def score(pred):
        w, y = tt.weight, tt.label; tp = (w * ((y == 1) & pred)).sum(); fp = (w * ((y == 0) & pred)).sum(); fn = (w * ((y == 1) & ~pred)).sum()
        return round(tp / (tp + fp), 3), round(tp / (tp + fn), 3), round(2 * tp / (2 * tp + fp + fn), 3)
    print("layer-1 rn5 on test set      P R F1:", score(tt.pred == 1))
    for t in (0.5, 0.6, 0.7, 0.8):
        print(f"rn5 minus rice (p >= {t})     P R F1:", score((tt.pred == 1) & (tt.p_rice < t)))


def apply(src, out, sites):
    J = joblib.load(L2 / "rice_model.joblib"); cols = J["columns"]
    for s in sites:
        s = SP.site_dir(s); d = s / "data/model"; o = d / out; o.mkdir(parents=True, exist_ok=True)
        with rasterio.open(d / src / "cane_map_2026.tif") as r: m, prof = r.read(1), r.profile.copy()
        with rasterio.open(s / "data/features/pheno_2026.tif") as r:
            names = list(r.descriptions); X = r.read().reshape(r.count, -1).astype("float32")
        X[X <= -32000] = np.nan
        unscaled = [i for i, n in enumerate(names) if n.split("_", 2)[-1] in ("last_cut", "grow_len", "grow_total") or n in ("greenup_days", "n_clear_mayoct")]
        X /= 100.0; X[unscaled] *= 100.0
        order = [names.index(c[4:]) for c in cols]
        idx = np.flatnonzero(m.ravel() == 1)
        p = np.full(m.size, 255, "uint8"); p[idx] = np.round(100 * J["model"].predict_proba(X[order][:, idx].T)[:, 1]).astype("uint8")
        p = p.reshape(m.shape); rice = (m == 1) & (p != 255) & (p >= 100 * J["thr"])
        cls = np.where(m == 255, 255, np.where(rice, 7, np.where(m == 1, 1, 0))).astype("uint8")
        new = np.where(rice, 0, m).astype("uint8")
        pr = dict(prof, count=1, dtype="uint8", nodata=255, compress="deflate", predictor=2)
        for name, a in (("cane_map_2026", new), ("class_2026", cls), ("rice_prob_2026", p)):
            with rasterio.open(o / f"{name}.tif", "w", **pr) as dst: dst.write(a, 1)
        Q.paletted(o / "cane_map_2026.tif", Q.MAP_CLASSES); Q.pseudocolor(o / "rice_prob_2026.tif", Q.PROB_STOPS)
        Q.paletted(o / "class_2026.tif", {0: ("#e6e6e6", "non-cane"), 1: ("#33ef2a", "sugarcane-like (cane + cassava)"), 7: ("#1B5E20", "rice (removed from cane)")})
        res = {"site": s.name, "src": src, "rice_thr": J["thr"], "cane_ac_before": round(float((m == 1).sum() * ACRE)), "rice_removed_ac": round(float(rice.sum() * ACRE)), "cane_ac_after": round(float((new == 1).sum() * ACRE))}
        (o / "layer2.json").write_text(json.dumps(res, indent=1)); print(res, flush=True)


if __name__ == "__main__":
    train() if sys.argv[1] == "train" else apply(sys.argv[2], sys.argv[3], sys.argv[4:])

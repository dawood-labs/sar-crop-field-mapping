"""Class separability check for the static 5-3-2 colour model, cane vs non-cane (user, 21 Oct 2026).
Input : sites/user_labels/colour_training_polygons_v1.gpkg (all user polygons: site, label, subtype, ref_date)
Samples: every polygon on every clear date >= 10 Sep (as the v8 training did), features = colour_multi.feats (8 colour features).
Tests : (1) grouped 5-fold CV by polygon, (2) leave-one-site-out; LightGBM cane vs non-cane; pixel- and polygon-level accuracy,
        confusion by subtype and site; list of polygons that fail. (3) feature medians per subtype.
Output: sites/layer2/colour_separability_v1/{samples.parquet, cv_polygons.csv, report.txt, scatter.png}"""
import sys, os, warnings
import site_paths as SP
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, lightgbm as lgb
from rasterio.features import rasterize
from rasterio.windows import Window
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent)); os.environ.setdefault("COLOUR_TAG", "v8")
import colour_multi as C
EXTRA = [b for b in os.environ.get("EXTRA_BANDS", "").split(",") if b]          # e.g. B4 or B4,B8 (21 Oct: user asked to test B4)
TAG = os.environ.get("SEP_TAG", "v1")
P = C.P; O = P / f"sites/layer2/colour_separability_{TAG}"; O.mkdir(parents=True, exist_ok=True); START = "20260910"
FCOLS = list(C.COLS) + (["B4", "B5/B4", "B4/B3"] if "B4" in EXTRA else []) + (["B8", "NDVI", "NDRE"] if "B8" in EXTRA else [])
D = gpd.read_file(P / "sites/user_labels/colour_training_polygons_v1.gpkg").to_crs(32648)
rows = []
for sd, pl in D.groupby("site"):
    st = SP.site_dir(sd) / "data/stacks"
    with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions); tr = c.transform
    late = [i for i, d in enumerate(dates) if d >= START]
    src = {b: rasterio.open(st / f"S2_{b}.tif") for b in ("B2", "B3", "B5", "cs_cdf", *EXTRA)}
    for _, q in pl.iterrows():
        x0, y0, x1, y1 = q.geometry.bounds; c0, r0 = ~tr * (x0, y1); c1, r1 = ~tr * (x1, y0)
        w = Window(int(c0), int(r0), max(1, int(np.ceil(c1)) - int(c0)), max(1, int(np.ceil(r1)) - int(r0))); wt = rasterio.windows.transform(w, tr)
        mk = rasterize([(q.geometry, 1)], out_shape=(int(w.height), int(w.width)), transform=wt, dtype="uint8").astype(bool)
        cs = src["cs_cdf"].read([i + 1 for i in late], window=w); b2 = src["B2"].read([i + 1 for i in late], window=w); b3 = src["B3"].read([i + 1 for i in late], window=w); b5 = src["B5"].read([i + 1 for i in late], window=w)
        ex = {b: src[b].read([i + 1 for i in late], window=w) for b in EXTRA}
        for k, i in enumerate(late):
            ok = mk & (cs[k] >= 60) & (cs[k] <= 100) & (b3[k] > 0)
            if ok.sum() < max(3, 0.6 * mk.sum()): continue
            X = pd.DataFrame(C.feats(b2[k][ok] / 1e4, b3[k][ok] / 1e4, b5[k][ok] / 1e4), columns=C.COLS)
            if "B4" in EXTRA: b4 = ex["B4"][k][ok] / 1e4; X["B4"] = b4; X["B5/B4"] = b5[k][ok] / 1e4 / (b4 + 1e-6); X["B4/B3"] = b4 / (b3[k][ok] / 1e4 + 1e-6)
            if "B8" in EXTRA: b8 = ex["B8"][k][ok] / 1e4; b4_ = ex["B4"][k][ok] / 1e4 if "B4" in EXTRA else None; X["B8"] = b8; X["NDVI"] = (b8 - b4_) / (b8 + b4_ + 1e-6) if b4_ is not None else np.nan; X["NDRE"] = (b8 - b5[k][ok] / 1e4) / (b8 + b5[k][ok] / 1e4 + 1e-6)
            rows.append(X.assign(y=int(q.label == "cane"), pid=q.poly_id, site=sd, subtype=q.subtype, date=dates[i]))
    for s in src.values(): s.close()
S = pd.concat(rows, ignore_index=True); S.to_parquet(O / "samples.parquet", index=False)
out = []
out.append(f"samples: {len(S):,} px from {S.pid.nunique()} polygons, {S.date.nunique()} image-dates, sites {sorted(S.site.unique())}")
out.append("px per class/subtype: " + str(S.groupby(["y", "subtype"]).size().to_dict()))
out.append("\n== feature medians by subtype ==\n" + S.groupby("subtype")[FCOLS].median().round(3).to_string())
def evaluate(name, folds):
    p = np.full(len(S), np.nan)
    for tr_i, te_i in folds:
        m = lgb.LGBMClassifier(**C.PARAMS).fit(S.iloc[tr_i][FCOLS], S.iloc[tr_i].y); p[te_i] = m.predict_proba(S.iloc[te_i][FCOLS])[:, 1]
    S[f"p_{name}"] = p; pol = S.groupby("pid").agg(y=("y", "first"), site=("site", "first"), subtype=("subtype", "first"), p=(f"p_{name}", "mean"), n=("y", "size"))
    pol["pred"] = (pol.p >= 0.5).astype(int); acc_px = float(((p >= 0.5).astype(int) == S.y).mean()); acc_pol = float((pol.pred == pol.y).mean())
    out.append(f"\n== {name}: pixel acc {acc_px:.3f} | polygon acc {acc_pol:.3f} ({len(pol)} polygons) ==")
    out.append("polygon-level by subtype (wrong / total):\n" + pol.groupby("subtype").apply(lambda g: f"{int((g.pred != g.y).sum())}/{len(g)}").to_string())
    out.append("polygon-level by site (wrong / total):\n" + pol.groupby("site").apply(lambda g: f"{int((g.pred != g.y).sum())}/{len(g)}").to_string())
    bad = pol[pol.pred != pol.y].sort_values("p"); out.append("failing polygons (poly_id: site subtype label p):\n" + "\n".join(f"  {i}: {r.site} {r.subtype} {'cane' if r.y else 'non-cane'} p={r.p:.2f}" for i, r in bad.iterrows()))
    return pol
gkf = GroupKFold(5); pol_cv = evaluate("cv5_by_polygon", list(gkf.split(S, S.y, S.pid)))
loso = [(np.flatnonzero(S.site != s), np.flatnonzero(S.site == s)) for s in sorted(S.site.unique())]; pol_lo = evaluate("leave_one_site_out", loso)
pol_cv.join(pol_lo[["p"]].rename(columns={"p": "p_loso"})).to_csv(O / "cv_polygons.csv")
(O / "report.txt").write_text("\n".join(out)); print("\n".join(out))
# scatter: hue vs B5 by subtype
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5)); cols = {"sugarcane": "#33ef2a", "cassava": "#8b1a1a", "rice": "#1B5E20", "other_tall_crop": "#ff8c00", "unspecified": "#555555", "unclear": "#999999"}
for sub, g in S.groupby("subtype"):
    g2 = g.sample(min(len(g), 3000), random_state=0); ax[0].scatter(g2.hue, g2.B5, s=3, alpha=.35, c=cols.get(sub, "k"), label=f"{sub} ({len(g):,})"); ax[1].scatter(g2["B5/B3"], g2.val, s=3, alpha=.35, c=cols.get(sub, "k"))
ax[0].set_xlabel("5-3-2 hue (deg)"); ax[0].set_ylabel("B5 reflectance"); ax[0].legend(markerscale=4, fontsize=8); ax[1].set_xlabel("B5/B3"); ax[1].set_ylabel("value (brightness)"); fig.suptitle("User polygons, all clear dates >= 10 Sep 2026: colour features by class")
fig.tight_layout(); fig.savefig(O / "scatter.png", dpi=110); print("wrote", O)

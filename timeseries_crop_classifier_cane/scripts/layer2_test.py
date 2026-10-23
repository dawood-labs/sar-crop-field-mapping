"""Layer 2 quick test: sugarcane vs cassava vs rice (vs other crop) inside the cane-like mask, from sites/layer2/points.parquet.
Leave-one-district-out CV on the review blocks; the test-set blocks are an extra holdout (never trained on).
Feature groups: sar (S1 Sep-Oct composite), opt (S2 latest-clear composite), sea (season / phenology), and combinations.
usage: python scripts/layer2_test.py   -> prints per-group / per-class precision, recall; sites/layer2/cv_results.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
P = Path("sites/layer2"); df = pd.read_parquet(P / "points.parquet")
CLASSES = ["sugarcane", "cassava", "rice", "other_crop"]
df = df[df.what.isin(CLASSES)].copy(); df["y"] = df.what.map({c: i for i, c in enumerate(CLASSES)})
tr = df[df.split == "train"]; te = df[df.split == "test"]
print("train:", tr.what.value_counts().to_dict(), " test:", te.what.value_counts().to_dict())
groups = {"sar": [c for c in df.columns if c.startswith("sar_")], "opt": [c for c in df.columns if c.startswith("opt_")],
          "sea": [c for c in df.columns if c.startswith("sea_")]}
groups["opx"] = [c for c in df.columns if c.startswith("opx_")]; groups["opt+opx"] = groups["opt"] + groups["opx"]
groups["opx+sea"] = groups["opx"] + groups["sea"]; groups["sar+opt"] = groups["sar"] + groups["opt"]; groups["opt+sea"] = groups["opt"] + groups["sea"]; groups["all"] = groups["sar"] + groups["opt"] + groups["opx"] + groups["sea"]
params = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, class_weight="balanced", verbose=-1, random_state=0)
rows = []
for gname, cols in groups.items():
    oof = np.full(len(tr), -1)
    for site in tr.site.unique():
        m = (tr.site == site).values
        mdl = lgb.LGBMClassifier(**params).fit(tr.loc[~m, cols], tr.y[~m]); oof[m] = mdl.predict(tr.loc[m, cols])
    mdl = lgb.LGBMClassifier(**params).fit(tr[cols], tr.y); pt = mdl.predict(te[cols])
    for name, y, yh in (("LODO-cv", tr.y.values, oof), ("testset", te.y.values, pt)):
        p, r, f, n = precision_recall_fscore_support(y, yh, labels=range(len(CLASSES)), zero_division=0)
        row = {"features": gname, "eval": name, "bal_acc": round(balanced_accuracy_score(y, yh), 3)}
        for i, c in enumerate(CLASSES): row[f"{c}_P"] = round(p[i], 2); row[f"{c}_R"] = round(r[i], 2); row[f"{c}_n"] = int(n[i])
        rows.append(row)
    if gname == "all":
        imp = pd.Series(mdl.feature_importances_, index=cols).sort_values(ascending=False); print("top features (all):", imp.head(12).to_dict())
res = pd.DataFrame(rows); res.to_csv(P / "cv_results.csv", index=False)
pd.set_option("display.width", 250); print(res.to_string(index=False))

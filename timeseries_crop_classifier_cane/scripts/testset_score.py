"""Score a map version on testset_v1 (blind reviews A, B and third reviewer C).

Usage: python scripts/testset_score.py [model_dir_name]   (default s2on_r2_oct)
Final label per block = majority of the non-unsure votes (A, B, C); blocks without a majority are dropped.
Writes sites/testset_v1/merge/final_labels.csv and sites/testset_v1/score_<model>.csv / .json.
Area-weighted with the key's stratum weights (pixels per sample).
"""
import glob, json, os, sys
import numpy as np, pandas as pd, rasterio
import site_paths as SP

ROOT = "."
TS = f"{ROOT}/sites/testset_v1"
model = sys.argv[1] if len(sys.argv) > 1 else "s2on_r2_oct"

m = pd.read_csv(f"{TS}/merge/ab_merged.csv").set_index("sample_id")
C = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(f"{TS}/labels/C_batch_*.csv"))])
C["sample_id"] = C.sample_id.str.strip(); C = C.set_index("sample_id")
m["answer_C"] = C["answer"].str.strip().str.lower().reindex(m.index)
m["what_C"] = C["what"].reindex(m.index)
need = m[m.third_needed & m.answer_C.isna()].index.tolist()
if need: print("WARNING: third review missing for", len(need), need[:10])

def final(r):
    v = [a for a in (r.answer_A, r.answer_B, r.answer_C) if a in ("cane", "not_cane")]
    c, n = v.count("cane"), v.count("not_cane")
    if c > n: return 1
    if n > c: return 0
    return -1
m["label"] = m.apply(final, axis=1)
m["votes"] = m[["answer_A", "answer_B", "answer_C"]].apply(lambda r: "/".join(str(x) for x in r), axis=1)
m[["answer_A", "answer_B", "answer_C", "what_A", "what_B", "what_C", "label", "votes", "reason"]].to_csv(f"{TS}/merge/final_labels.csv")

k = pd.read_csv(f"{TS}/key/key.csv").set_index("sample_id")
SITES = [x for x in os.environ.get("SITES", "").split(",") if x]          # optional: score only these sites
if SITES: k = k[k.site.isin(SITES)]
k["label"] = m["label"]
pred = {}
for site, g in k.groupby("site"):
    with rasterio.open(f"{SP.site_dir(site)}/data/model/{model}/cane_map_2026.tif") as src:
        a = src.read(1)
    for sid, r in g.iterrows(): pred[sid] = int(a[int(r.row), int(r.col)] == 1)
k["pred"] = pd.Series(pred)
k.to_csv(f"{TS}/score_{model}_rows.csv")
use = k[k.label >= 0].copy()
print(f"blocks used {len(use)} / {len(k)} (dropped {int((k.label<0).sum())} without majority)")

def score(d):
    w = d.weight; y = d.label; p = d.pred
    tp = (w*((y==1)&(p==1))).sum(); fp = (w*((y==0)&(p==1))).sum(); fn = (w*((y==1)&(p==0))).sum()
    P = tp/(tp+fp) if tp+fp else np.nan; R = tp/(tp+fn) if tp+fn else np.nan
    F = 2*P*R/(P+R) if P+R else np.nan
    n_tp=int(((y==1)&(p==1)).sum()); n_fp=int(((y==0)&(p==1)).sum()); n_fn=int(((y==1)&(p==0)).sum()); n_tn=int(((y==0)&(p==0)).sum())
    acc = (w*(y==p)).sum()/w.sum()
    return dict(n=len(d), tp=n_tp, fp=n_fp, fn=n_fn, tn=n_tn, acc_w=round(acc,3), P_w=round(P,3), R_w=round(R,3), F1_w=round(F,3),
                map_cane_ac=round((w*p).sum()*0.0247105,0), est_true_cane_ac=round((w*y).sum()*0.0247105,0))
rows = {"ALL": score(use)}
for site, g in use.groupby("site"): rows[site.replace("_2026","")] = score(g)
out = pd.DataFrame(rows).T
out.to_csv(f"{TS}/score_{model}{'_' + '_'.join(SITES) if SITES else ''}.csv")
print(out.to_string())
print("\nby stratum:"); print(pd.DataFrame({s: score(g) for s, g in use.groupby("stratum")}).T[["n","tp","fp","fn","tn","acc_w"]].to_string())

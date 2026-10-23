"""Check the rice-in-cane output (model/rice_cane/rice_2026.tif) against the user's polygons, the test set and totals.
usage: python scripts/rice_cane_check.py   -> prints tables, writes sites/layer2/rice_cane_summary.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.features import rasterize
import site_paths as SP
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.argv = ["x"]; import rice_cane as R
P = R.P; ACRE = R.ACRE; pd.set_option("display.width", 250)
rows = []; up = R.user_polys(); pol = []
key = pd.read_csv(P / "sites/testset_v1/key/key.csv").merge(pd.read_csv(P / "sites/testset_v1/merge/final_labels.csv"), on="sample_id"); ts = []
for d in sorted(P.glob("sites/20??/*/*_2026/data/model/rice_cane/rice.json")):
    s = d.parents[3].name; j = json.load(open(d)); S = SP.site_dir(s) / "data"
    with rasterio.open(S / "model/rice_cane/rice_2026.tif") as r: rice, tr, shp = r.read(1), r.transform, r.shape
    with rasterio.open(S / "model/layer2_final/class3_2026.tif") as r: cls = r.read(1)
    rows.append(dict(site=s.replace("_2026", ""), sugarcane_class_ac=j["sugarcane_class_ac"], rice_out_ac=j["rice_ac"], rice_pct=round(100 * j["rice_ac"] / max(j["sugarcane_class_ac"], 1), 1), sugarcane_final_ac=j["sugarcane_final_ac"]))
    for _, q in up[up.site == s].iterrows():
        m = rasterize([(q.geometry, 1)], out_shape=shp, transform=tr, dtype="uint8").astype(bool); ins = m & (cls == 1)
        pol.append(dict(site=s.replace("_2026", ""), label="rice" if q.y else "cane", px=int(m.sum()), in_sugarcane_px=int(ins.sum()), flagged_rice_px=int((rice[ins] == 1).sum())))
    for _, q in key[key.site == s].iterrows():
        blk = cls[q.row - 1:q.row + 2, q.col - 1:q.col + 2]; rb = rice[q.row - 1:q.row + 2, q.col - 1:q.col + 2]
        if (blk == 1).sum() == 0: continue
        w = (str(q.what_A), str(q.what_B), str(q.what_C))
        ts.append(dict(site=s, sample_id=q.sample_id, label=("cane" if q.label == 1 else ("rice" if "rice" in w else "other_not_cane")), weight=q.weight, frac_rice=float((rb[blk == 1] == 1).mean())))
T = pd.DataFrame(rows); T.loc[len(T)] = dict(site="TOTAL", sugarcane_class_ac=T.sugarcane_class_ac.sum(), rice_out_ac=T.rice_out_ac.sum(), rice_pct=round(100 * T.rice_out_ac.sum() / T.sugarcane_class_ac.sum(), 1), sugarcane_final_ac=T.sugarcane_final_ac.sum())
print("=== per district (acres) ==="); print(T.to_string(index=False)); T.to_csv(P / "sites/layer2/rice_cane_summary.csv", index=False)
Pp = pd.DataFrame(pol)
if len(Pp):
    g = Pp.groupby(["site", "label"]).agg(polygons=("px", "size"), px=("px", "sum"), in_sugarcane_px=("in_sugarcane_px", "sum"), flagged_rice_px=("flagged_rice_px", "sum")).reset_index()
    g["flagged_pct_of_in_class"] = (100 * g.flagged_rice_px / g.in_sugarcane_px.replace(0, np.nan)).round(0)
    print("\n=== user polygons: pixels inside the sugarcane class and how many the model flags as rice ==="); print(g.to_string(index=False))
    Pp["caught"] = Pp.flagged_rice_px / Pp.in_sugarcane_px.replace(0, np.nan)
    pr = Pp[(Pp.label == "rice") & (Pp.in_sugarcane_px >= 5)]; print(f"rice polygons with >= 5 px in class: {len(pr)}; >= 50 % flagged: {int((pr.caught >= .5).sum())}; 0 % flagged: {int((pr.caught == 0).sum())}")
    pc = Pp[(Pp.label == "cane") & (Pp.in_sugarcane_px >= 5)]; print(f"cane polygons with >= 5 px in class: {len(pc)}; any px flagged: {int((pc.caught > 0).sum())}; >= 50 % flagged: {int((pc.caught >= .5).sum())}")
Tt = pd.DataFrame(ts)
if len(Tt):
    print("\n=== test-set blocks inside the sugarcane class (block-level, 3x3) ===")
    for lab, g in Tt.groupby("label"): print(f"{lab:15s} blocks {len(g):4d}  flagged >= 50 %: {int((g.frac_rice >= .5).sum()):3d}  ({100 * (g.frac_rice >= .5).mean():.0f} %)  area-weighted {100 * np.average(g.frac_rice >= .5, weights=g.weight):.1f} %")

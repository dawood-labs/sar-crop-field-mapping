"""Compare the blind analyst labels (audit/labels/batch_*.csv) with rice model v1 (audit/key/audit_key.csv).

Outputs in data/model/rice_v1/audit/: audit_results.csv (+ .gpkg for QGIS), audit_summary.json; prints agreement per
probability bin, the error types (analyst subtype where model and analyst disagree) and the disagreement list.
"""
from __future__ import annotations

from pathlib import Path
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
VER = sys.argv[1] if len(sys.argv) > 1 else "rice_v1"          # usage: audit_compare.py [rice_v1|rice_v2]
A = PROJ / "data/model" / VER / "audit"


def main():
    key = pd.read_csv(A / "key/audit_key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((A / "labels").glob("batch_*.csv"))], ignore_index=True)
    lab["audit_id"] = lab.audit_id.str.strip()
    for c in ["label", "subtype", "confidence"]:
        lab[c] = lab[c].astype(str).str.strip()
    d = key.merge(lab, on="audit_id", how="left")
    missing = d.label.isna().sum()
    d["agree"] = (d.label == d.model_class)
    d.loc[d.label == "Unsure", "agree"] = pd.NA
    order = ["rice_high", "rice_med", "rice_low", "non_near", "non_med", "non_low"]
    t = d.groupby("bin").agg(n=("audit_id", "size"),
                             analyst_rice=("label", lambda s: (s == "Rice").sum()),
                             analyst_nonrice=("label", lambda s: (s == "Non-rice").sum()),
                             unsure=("label", lambda s: (s == "Unsure").sum()),
                             agree=("agree", lambda s: int(s.fillna(False).astype(bool).sum())))
    t = t.reindex(order)
    t["agreement_%_of_decided"] = (100 * t.agree / (t.n - t.unsure)).round(0)
    print(t.to_string())
    dec = d[d.label.isin(["Rice", "Non-rice"])]
    tp = ((dec.model_class == "Rice") & (dec.label == "Rice")).sum()
    fp = ((dec.model_class == "Rice") & (dec.label == "Non-rice")).sum()
    fn = ((dec.model_class == "Non-rice") & (dec.label == "Rice")).sum()
    tn = ((dec.model_class == "Non-rice") & (dec.label == "Non-rice")).sum()
    s = {"n": int(len(d)), "missing_labels": int(missing), "unsure": int((d.label == "Unsure").sum()),
         "confusion_model_vs_analyst": {"rice_rice": int(tp), "rice_but_analyst_non": int(fp),
                                        "non_but_analyst_rice": int(fn), "non_non": int(tn)},
         "precision_vs_analyst_sample": round(tp / max(tp + fp, 1), 3), "recall_vs_analyst_sample": round(tp / max(tp + fn, 1), 3),
         "note": "sample is stratified by probability bin, NOT proportional to the map: use per-bin rates, not these totals, for map-wide accuracy"}
    print(json.dumps(s, indent=1))
    print("\nwhat the model calls rice but the analyst does not (subtype):",
          d[(d.model_class == "Rice") & (d.label == "Non-rice")].subtype.value_counts().to_dict())
    print("what the model calls non-rice but the analyst calls rice, by bin:",
          d[(d.model_class == "Non-rice") & (d.label == "Rice")].bin.value_counts().to_dict())
    dis = d[(d.agree == False) | (d.label == "Unsure")].sort_values(["bin", "audit_id"])  # noqa: E712
    pd.set_option("display.max_colwidth", 140); pd.set_option("display.width", 250)
    print(dis[["audit_id", "bin", "model_prob", "label", "subtype", "confidence", "evidence"]].to_string(index=False))
    # map-wide estimate: weight each bin by its share of the map (the sample itself is stratified, not proportional)
    import rasterio
    prob = rasterio.open(A.parent / "rice_prob.tif").read(1)
    ins = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    v = prob[ins & (prob != 255)]
    lims = {"rice_high": (90, 100), "rice_med": (70, 89), "rice_low": (50, 69), "non_near": (31, 49), "non_med": (11, 30), "non_low": (0, 10)}
    acre = 100 / 4046.856
    rows = []
    for b, (lo, hi) in lims.items():
        n = int(((v >= lo) & (v <= hi)).sum()); q = dec[dec.bin == b]
        fr = float((q.label == "Rice").mean()) if len(q) else np.nan
        rows.append(dict(bin=b, map_acres=round(n * acre), n_decided=len(q), analyst_rice_share=round(fr, 3), est_rice_acres=round(n * acre * fr)))
    w = pd.DataFrame(rows); print(w.to_string(index=False))
    mr = w[w.bin.str.startswith("rice")]
    est = {"map_rice_acres": int(mr.map_acres.sum()), "audit_based_rice_acres": int(w.est_rice_acres.sum()),
           "area_weighted_precision": round(mr.est_rice_acres.sum() / mr.map_acres.sum(), 3),
           "area_weighted_recall": round(mr.est_rice_acres.sum() / w.est_rice_acres.sum(), 3)}
    print(est); s["area_weighted"] = est; s["per_bin_area"] = rows
    d.to_csv(A / "audit_results.csv", index=False)
    (A / "audit_summary.json").write_text(json.dumps({**s, "per_bin": t.reset_index().to_dict("records")}, indent=1, default=int))
    import geopandas as gpd
    gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326).to_file(A / "audit_results.gpkg", driver="GPKG")


if __name__ == "__main__":
    main()

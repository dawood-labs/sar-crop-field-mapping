"""Blind review of the S2 plant-cane RECALL RULE (round 3): blocks the rule added that the model map had missed.

  python scripts/rule_review.py sample [src tag]   -> sites/rule_review_<tag>/ (sheets/R###.png, RUBRIC.md, key/, labels/)
  python scripts/rule_review.py compare [src tag]  -> review_results.csv (+ precision of the rule, area-weighted)
Eligible 3 x 3 blocks: cane_source == 2 (rule-added) over the full 5 x 5 window (interior of a rule patch), the 2025 label
one class over 3 x 3, S2 uniform (indep_review.eligible rule). N_PER_AOI blocks per AOI at random, >= 300 m from the
test set and earlier reviews (EXCLUDE_FROM). Sheets and rubric = scripts/indep_review.py (reviewers never see map or key).
Answers feed round-3 training as hard examples (cane = 1, not_cane = 0); the test set is never used for training.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
TAG = sys.argv[2] if len(sys.argv) > 2 else "s2on_r2_oct_rn4"
sys.argv = [sys.argv[0], sys.argv[1], TAG]
os.environ.setdefault("OUT_NAME", f"rule_review_{TAG}")
os.environ.setdefault("ID_PREFIX", "R")
import indep_review as IR  # noqa: E402

import numpy as np, pandas as pd, rasterio  # noqa: E402
import site_paths as SP

N_PER_AOI = int(os.environ.get("N_PER_AOI", "10"))
AOIS = [a for a in IR.AOIS if a not in os.environ.get("SKIP_AOIS", "").split(",")]


def sample():
    from scipy import ndimage
    for x in ("sheets", "key", "labels", "google"):
        (IR.OUT / x).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(IR.SEED)
    rows = []
    for s in AOIS:
        d = SP.site_dir(s) / "data"
        src = rasterio.open(d / f"model/{TAG}/cane_source_2026.tif").read(1)
        m, lab, ok = IR.eligible(s)                      # map uniform 5x5 (cane after the rule), label uniform, S2 uniform
        k5 = np.ones((5, 5), bool)
        rule5 = (ndimage.minimum_filter((src == 2).astype("uint8"), footprint=k5) == 1)
        ok &= rule5
        for f in IR.EXCLUDE_FROM:
            k = pd.read_csv(f); k = k[k.site == s]
            for r_, c_ in zip(k.row, k.col):
                ok[max(r_ - 30, 0):r_ + 31, max(c_ - 30, 0):c_ + 31] = False
        rr, cc = np.nonzero(ok)
        k = rng.choice(len(rr), min(N_PER_AOI, len(rr)), replace=False)
        n_rule = int((src == 2).sum())
        for r, c in zip(rr[k], cc[k]):
            rows.append({"site": s, "stratum": "rule_added", "row": int(r), "col": int(c), "pixel_id": int(r * m.shape[1] + c),
                         "label2025": int(lab[r, c]), "map_tag": 1, "n_stratum_px": n_rule, "n_eligible_px": len(rr),
                         "weight": n_rule / max(len(k), 1)})
        print(f"{s}: rule-added px {n_rule:,} ({n_rule * 0.0247105:,.0f} ac), eligible interior {len(rr):,}, sampled {len(k)}", flush=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=IR.SEED).reset_index(drop=True)
    df["sample_id"] = [f"{os.environ['ID_PREFIX']}{i + 1:03d}" for i in range(len(df))]
    import geopandas as gpd
    tr = {s: rasterio.open(SP.site_dir(s) / "data/model" / TAG / "cane_map_2026.tif").transform for s in AOIS}
    x = [tr[r.site].c + (r.col + .5) * 10 for r in df.itertuples()]; y = [tr[r.site].f - (r.row + .5) * 10 for r in df.itertuples()]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=32648)
    ll = g.to_crs(4326).geometry; df["lon"], df["lat"] = ll.x.round(6), ll.y.round(6); g["lon"], g["lat"] = df.lon, df.lat
    df.to_csv(IR.OUT / "key/key.csv", index=False); g.to_parquet(IR.OUT / "key/samples.parquet")
    (IR.OUT / "RUBRIC.md").write_text(IR.RUBRIC)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(r.site, r.sample_id, r.row, r.col, r.lon, r.lat) for r in df.itertuples()]
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(IR.sheet, jobs, chunksize=2))
    print(df.groupby("site").size().to_string(), f"\n{len(df)} sheets -> {IR.OUT / 'sheets'}")


def compare():
    key = pd.read_csv(IR.OUT / "key/key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((IR.OUT / "labels").glob("*.csv"))], ignore_index=True)
    d = key.merge(lab[["sample_id", "answer", "what", "confidence", "note"]], on="sample_id", how="left")
    d["y"] = d.answer.map({"cane": 1, "not_cane": 0})
    d.to_csv(IR.OUT / "review_results.csv", index=False)
    x = d[d.y.notna()]
    print(d.answer.value_counts().to_dict()); print(pd.crosstab(d.site, d.what).to_string())
    for site in AOIS + ["ALL"]:
        z = x[(x.site == site) | (site == "ALL")]
        if len(z):
            print(f"{site:24s} n {len(z):3d}  rule precision (area-weighted) {(z.weight * z.y).sum() / z.weight.sum():.2f}")


if __name__ == "__main__":
    {"sample": sample, "compare": compare}[sys.argv[1]]()

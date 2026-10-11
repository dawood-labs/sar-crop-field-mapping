"""Blind audit of rice v4 by FIELD: 400 pixels, 200 mapped rice + 200 mapped non-rice, one pixel per FTW field, never on
a field edge. Goal: new, clean training / test labels to improve the rice map (user, 2026-10-09).

Sampling (data/model/rice_v4/rice_prob.tif, data/fields/ftw_field_id.tif):
  - a pixel is eligible only if its 3x3 neighbourhood (= the SAR feature window) lies in the same FTW field, i.e. it
    does not touch the field boundary (FTW paddies are small: median 0.37 acre, so 5x5 left too few fields);
  - fields holding any earlier label (GCP, audits v1-v3, labels_v1 candidates, v4 training samples) are skipped, and new
    pixels are >= 100 m from every earlier label and >= 200 m from each other;
  - one random interior pixel per field; its v4 probability puts it in a bin:
      rice: >=90 (70), 70-89 (70), 50-69 (60)      non-rice: <=10 (60), 11-30 (70), 31-49 (70)
  - IDs F001..F400 are shuffled, so a batch says nothing about the model's answer.
Sheets are the same blind sheets as the earlier audits (audit_sheets.sheet). FTW is only a reference for field location.
Outputs (data/model/rice_v4/audit/): key/audit_key.csv (NOT for the analysts), points.csv, sheets/F###.png
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("GDAL_CACHEMAX", "256")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))
sys.argv = [sys.argv[0], "rice_v4"]                 # audit_sheets reads the model version from argv at import
import audit_sheets as asx  # noqa: E402
from safe_resources import workers  # noqa: E402

M, A = asx.M, asx.A
BINS = [("rice_high", 90, 100, 70), ("rice_med", 70, 89, 70), ("rice_low", 50, 69, 60),
        ("non_low", 0, 10, 60), ("non_med", 11, 30, 70), ("non_near", 31, 49, 70)]
SEED = 2026


def earlier_labels():
    import geopandas as gpd
    out = [pd.read_csv(PROJ / "data/gcps/eda/gcp_verdicts.csv")[["row", "col"]],
           pd.read_csv(PROJ / "data/labels_v1/key/candidates_key.csv")[["row", "col"]],
           gpd.read_file(M / "samples.gpkg")[["row", "col"]]]
    out += [pd.read_csv(k)[["row", "col"]] for k in (PROJ / "data/model").glob("rice_v[123]/audit/key/audit_key.csv")]
    return pd.concat(out).drop_duplicates().to_numpy()


def sample():
    from scipy import ndimage
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(SEED)
    prob = rasterio.open(M / "rice_prob.tif").read(1)
    fid = rasterio.open(PROJ / "data/fields/ftw_field_id.tif").read(1)
    inside = rasterio.open(PROJ / "data/grid/index.tif").read(2).astype(bool)
    interior = (fid > 0) & (ndimage.maximum_filter(fid, 3) == fid) & (ndimage.minimum_filter(fid, 3) == fid) & inside & (prob != 255)
    old = earlier_labels()
    bad_fields = set(np.unique(fid[old[:, 0], old[:, 1]])) - {0}
    rr, cc = np.nonzero(interior)
    f = fid[rr, cc]
    keep = ~np.isin(f, list(bad_fields)) & (cKDTree(old).query(np.c_[rr, cc])[0] >= 10)
    rr, cc, f = rr[keep], cc[keep], f[keep]
    order = rng.permutation(len(rr))                        # one random interior pixel per field
    _, first = np.unique(f[order], return_index=True)
    pick = order[first]
    c = pd.DataFrame({"row": rr[pick], "col": cc[pick], "field_id": f[pick]})
    c["n_interior"] = np.bincount(f)[c.field_id]
    c["model_prob"] = prob[c.row, c.col]
    print(f"eligible fields {len(c)} (skipped {len(bad_fields)} with earlier labels)", flush=True)
    rows, pts = [], []
    for name, lo, hi, n in BINS:
        q = c[(c.model_prob >= lo) & (c.model_prob <= hi)].sample(frac=1, random_state=int(rng.integers(1e9)))
        got = 0
        for r in q.itertuples():
            if pts and cKDTree(pts).query((r.row, r.col))[0] < 20:
                continue
            pts.append((r.row, r.col)); rows.append(dict(r._asdict(), bin=name)); got += 1
            if got >= n:
                break
        print(f"  {name}: {len(q)} fields -> {got}", flush=True)
    d = pd.DataFrame(rows).drop(columns="Index")
    d["model_class"] = np.where(d.model_prob >= 50, "Rice", "Non-rice")
    d["v3_prob"] = rasterio.open(PROJ / "data/model/rice_v3/rice_prob.tif").read(1)[d.row, d.col]
    d = d.sample(frac=1, random_state=7).reset_index(drop=True)
    d.insert(0, "audit_id", [f"F{i + 1:03d}" for i in range(len(d))])
    W, H, tr, crs = asx.ex._grid()
    from pyproj import Transformer
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    d["lon"], d["lat"] = t.transform((tr.c + (d.col + 0.5) * 10).values, (tr.f - (d.row + 0.5) * 10).values)
    return d


def main():
    (A / "sheets").mkdir(parents=True, exist_ok=True); (A / "key").mkdir(exist_ok=True); (A / "labels").mkdir(exist_ok=True)
    d = sample()
    d.to_csv(A / "key/audit_key.csv", index=False)
    d[["audit_id", "lon", "lat"]].to_csv(A / "points.csv", index=False)
    print(d.groupby("bin").size().to_dict(), "v3/v4 disagree:", int(((d.v3_prob >= 50) != (d.model_prob >= 50)).sum()), flush=True)
    T = asx.templates()
    from concurrent.futures import ProcessPoolExecutor
    n = workers(1.5 * 2**30)
    print(f"{n} sheet workers", flush=True)
    with ProcessPoolExecutor(n) as pool:
        list(pool.map(asx._draw, [d.iloc[i::n] for i in range(n)], [T] * n))
    print("sheets:", len(list((A / "sheets").glob("F*.png"))))


if __name__ == "__main__":
    main()

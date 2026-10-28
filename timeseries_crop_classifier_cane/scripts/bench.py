"""Fixed benchmark (user, 13 Oct 2026): districts NEVER used in training, a RANDOM stratified pixel sample reviewed
blind once, and every model round scored on the same pixels (area-weighted precision, recall, F1 and cane area).

  python scripts/bench.py sample   -> sites/benchmark/: sheets/B###.png (all sites), RUBRIC.md, key/key.csv (site,
                                      pixel, stratum, weight), user/user_review.csv (20 px for the user, blind)
  python scripts/bench.py score <model tag> [<model tag> ...]   -> sites/benchmark/scores.csv + printed table
Strata per district = the client's 2025 map (cane / non-cane inside the AOI), N_PER pixels each at random (edges
included: a real map has edges); weight = stratum pixels / sampled pixels -> unbiased estimates for the district.
Reviewer answers: cane (incl. cane-like cassava) = 1, not_cane = 0, unsure = left out (reported).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import site_paths as SP

PROJ = Path(__file__).resolve().parents[1]
SITES = ["bench_site102_2026", "bench_site26_2026"]
OUT = PROJ / "sites/benchmark"
N_PER, SEED, N_USER = 60, 42, 20


def sample():
    import site_review as SR
    for d in ("sheets", "key", "labels", "user"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for s in SITES:
        lab = rasterio.open(SP.site_dir(s) / "data/labels/cane2025_labels.tif").read(1)
        for cls, name in ((1, "label_cane"), (0, "label_noncane")):
            rr, cc = np.nonzero(lab == cls)
            k = rng.choice(len(rr), N_PER, replace=False)
            for r, c in zip(rr[k], cc[k]):
                rows.append({"site": s, "stratum": name, "row": int(r), "col": int(c), "pixel_id": int(r * lab.shape[1] + c),
                             "weight": len(rr) / N_PER})
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["sample_id"] = [f"B{i + 1:03d}" for i in range(len(df))]
    df.to_csv(OUT / "key/key.csv", index=False)
    (OUT / "RUBRIC.md").write_text((PROJ / "reports/review/RUBRIC_2026.md").read_text())
    # sheets (one process per core; each site's own stacks / features)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    os.environ["REVIEW_YEAR"] = "2026"
    SR.YEAR = 2026
    SR.S2_START, SR.S2_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-11-30")
    jobs = []
    for r in df.itertuples():
        site = SP.site_dir(r.site)
        jobs.append((str(site), {"sample_id": r.sample_id, "row": r.row, "col": r.col}, str(OUT)))
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_sheet, jobs, chunksize=4))
    # the user's blind subset: 10 per district, both strata
    u = df.groupby(["site", "stratum"], group_keys=False).apply(lambda g: g.sample(N_USER // 4, random_state=SEED))
    tr = {s: rasterio.open(SP.site_dir(s) / "data/grid/index.tif").transform for s in SITES}
    from pyproj import Transformer
    T = Transformer.from_crs(32648, 4326, always_xy=True)
    ll = [T.transform(tr[r.site].c + (r.col + .5) * 10, tr[r.site].f - (r.row + .5) * 10) for r in u.itertuples()]
    u = u.assign(lon=[round(a, 6) for a, _ in ll], lat=[round(b, 6) for _, b in ll])
    u[["sample_id", "site", "pixel_id", "lon", "lat"]].assign(your_answer="", your_what="", your_note="") \
        .sort_values("sample_id").to_csv(OUT / "user/user_review.csv", index=False)
    print(df.groupby(["site", "stratum"]).size().to_string(), f"\n{len(df)} sheets -> {OUT / 'sheets'}")


def _sheet(args):
    import os as _os
    _os.environ["REVIEW_YEAR"] = "2026"
    import site_review as SR
    SR.YEAR = 2026
    SR.S2_START, SR.S2_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-11-30")
    site, job, out = args
    # site_review.sheet writes into <site>/<folder>/sheets; render there then move to the benchmark folder
    tmp = Path(site) / "bench_tmp"
    (tmp / "sheets").mkdir(parents=True, exist_ok=True)
    SR.sheet((site, job, "bench_tmp"))
    src = tmp / "sheets" / f"{job['sample_id']}.png"
    src.replace(Path(out) / "sheets" / src.name)


def truth():
    key = pd.read_csv(OUT / "key/key.csv")
    files = sorted((OUT / "labels").glob("batch_*.csv"))
    lab = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = key.merge(lab[["sample_id", "answer", "what", "confidence"]], on="sample_id", how="left")
    d["y"] = d.answer.map({"cane": 1, "not_cane": 0})
    return d


def score(tags):
    d = truth()
    rows = []
    for tag in tags:
        p = np.full(len(d), np.nan)
        for s in SITES:
            f = SP.site_dir(s) / "data/model" / tag / "cane_prob_2026.tif"
            if not f.exists():
                continue
            a = rasterio.open(f).read(1)
            fm = f.with_name("cane_map_2026.tif")             # post-rules (e.g. October bare) live in the map
            if fm.exists():
                mp = rasterio.open(fm).read(1)
                a = np.where(mp == 255, 255, np.where(mp == 1, np.maximum(a, 50), np.minimum(a, 49))).astype(a.dtype)
            m = (d.site == s).values
            p[m] = a[d.row[m], d.col[m]]
        d[f"p_{tag}"] = p
        for site in SITES + ["ALL"]:
            xa = d[((d.site == site) | (site == "ALL")) & (d[f"p_{tag}"] != 255) & d[f"p_{tag}"].notna()]
            x = xa[xa.y.notna()]                     # precision / recall on sure answers; map area on every pixel
            yh, y, w = (x[f"p_{tag}"] >= 50).astype(int), x.y.astype(int), x.weight
            tp, fp, fn = (w * (yh & y)).sum(), (w * (yh & (1 - y))).sum(), (w * ((1 - yh) & y)).sum()
            P, R = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
            ac = 100 / 4046.8564224
            rows.append({"model": tag, "site": site, "n_sure": len(x), "precision": round(P, 3), "recall": round(R, 3),
                         "f1": round(2 * P * R / max(P + R, 1e-9), 3), "true_cane_acres_est": round((w * y).sum() * ac),
                         "map_cane_acres_est": round((xa.weight * (xa[f"p_{tag}"] >= 50)).sum() * ac), "oa": round(float((w * (yh == y)).sum() / w.sum()), 3)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "scores.csv", index=False)
    d.to_csv(OUT / "review_results.csv", index=False)
    print(f"answers: {d.answer.value_counts(dropna=False).to_dict()}")
    print(r.to_string(index=False))


if __name__ == "__main__":
    {"sample": lambda: sample(), "score": lambda: score(sys.argv[2:])}[sys.argv[1]]()

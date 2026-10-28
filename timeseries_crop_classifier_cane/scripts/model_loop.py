"""One round of the district loop (user, 13 Oct 2026): retrain on every district done so far, then test on new ones.

Training data per district (100k cane + 100k non-cane, clean label, edge >= 2):
  2025 sites: label = 2025 map after label QC on 2025 data
  2026 sites: label = 2025 map after label QC on 2026 data (changed fields drop out)
plus the blind-review answers of every reviewed 2026 district as hard examples (3 x 3 px around each reviewed pixel,
weight REVIEW_WEIGHT): answer cane (sugarcane or cane-like, e.g. cassava) -> 1, not_cane (medium / high confidence) -> 0,
unsure -> not used.
Features: VV + VH 15-day composites Jan - 10 Oct (2026 band names shifted to 2025); LightGBM defaults; threshold 0.5.
usage: TAG=loop1 SITES_2025=site99,site21,site01 \
       SITES_2026=site99_2026,site92_2026,site116_2026,site20_2026 python scripts/model_loop.py
Output: data/model/<TAG>/model.joblib (same format as the other models) + train.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402

import json, os, time  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

import model_2025 as M  # noqa: E402
import site_paths as SP

PROJ = M.PROJ
TAG = os.environ.get("TAG", "loop1")
OUT = PROJ / "data/model" / TAG
S25 = [s for s in os.environ.get("SITES_2025", "site99,site21,site01").split(",") if s]
S26 = [s for s in os.environ.get("SITES_2026", "site99_2026").split(",") if s]
FS = os.environ.get("FS", "VVVH@202501a")                 # Nov start test: VVVH@202411a
FEAT = os.environ.get("FEAT", "jan")                       # composites s1_15d_<year><FEAT>.tif ("jan" | "nov")
NEWCANE_EXCLUDE = os.environ.get("NEWCANE_EXCLUDE") == "1"  # 2026: drop non-cane px that look like Oct-Dec plantings
N_PER_CLASS, SEED = int(os.environ.get("N_PER_CLASS", "100000")), int(os.environ.get("SEED", "0"))   # label-derived px per class per site
REVIEW_WEIGHT = float(os.environ.get("REVIEW_WEIGHT", "3"))          # was 10 in loop1-3 (user 13 Oct: 3, sure answers only)
REVIEW_CONF = os.environ.get("REVIEW_CONF", "medium,high").split(",")  # answers used as hard examples
S2FEAT = os.environ.get("S2FEAT") == "1"                              # + dry-season S2 features (test)
PHENO = os.environ.get("PHENO") == "1"
PLANT_EXCLUDE = os.environ.get("PLANT_EXCLUDE") == "1"   # 2026: drop label non-cane px with a plant-cane pattern (bare dry season, green by May-Jun, dense Jul-Sep)                                # + date-free phenology features (pheno_features.py, round 3)
RATOON_EXCLUDE = os.environ.get("RATOON_EXCLUDE") == "1"      # 2026: also drop non-cane px with dry-season ratoon regrowth
EXTRA_REVIEW = [Path(x) for x in os.environ.get("EXTRA_REVIEW", "").split(",") if x]  # e.g. indep review results (column site)
SITE_W = {k: float(v) for k, v in (x.split(":") for x in os.environ.get("SITE_WEIGHTS", "").split(",") if x)}  # e.g. site66_2026:0.3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def paths(site, year):
    root = PROJ if site == "site99" else SP.site_dir(site)
    feat = root / "data/features" / ("s1_15d_2025.tif" if (site == "site99" and FEAT == "jan") else
                                     "s1_15d_2025nov.tif" if site == "site99" else f"s1_15d_{year}{FEAT}.tif")
    return root, feat


def read_feat(feat, year):
    with rasterio.open(feat) as r:
        names, X = list(r.descriptions), r.read()
    if year == 2026:
        names = [n[:-7] + str(int(n[-7:-3]) - 1) + n[-3:] for n in names]
    return names, X


def s2_block(root, year):
    """Dry-season S2 features (scripts/s2_dry_features.py) as float with NaN for no data, or None if absent."""
    f = root / f"data/features/s2_dry_{year}.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as r:
        a = r.read().reshape(r.count, -1).astype("float32")
        names = [n.replace(str(year - 1) + "12", "PREV12").replace(str(year), "Y") for n in r.descriptions]
    a[a <= -32000] = np.nan
    return a / 1000, names


def pheno_block(root, year):
    """Date-free phenology features (scripts/pheno_features.py) as float with NaN for no data, or None if absent."""
    f = root / f"data/features/pheno_{year}.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as r:
        a = r.read().reshape(r.count, -1).astype("float32"); names = list(r.descriptions)
    a[a <= -32000] = np.nan
    unscaled = [i for i, n in enumerate(names) if n.split("_", 2)[-1] in ("last_cut", "grow_len", "grow_total") or n in ("greenup_days", "n_clear_mayoct")]
    a /= 100.0
    a[unscaled] *= 100.0
    return a, ["PH_" + n for n in names]


def new_cane_mask(root, year):
    """S2 dry-season pattern of a field planted Oct-Dec of year-1: bare in Dec (NDVI < 0.30), growing through the dry
    season (Feb >= 0.40, Apr >= 0.50). Also matches dry-season irrigated crops, so it is used only to EXCLUDE these
    pixels from the non-cane class (the 2025 map, dated 29 Nov, saw them bare)."""
    b = s2_block(root, year)
    if b is None:
        return None
    a, n = b
    g = lambda k: a[n.index(k)].reshape(rasterio.open(root / "data/labels/cane2025_qc.tif").shape)  # noqa: E731
    with np.errstate(invalid="ignore"):
        m = (g("S2_NDVI_PREV12") < 0.30) & (g("S2_NDVI_Y02") >= 0.40) & (g("S2_NDVI_Y04") >= 0.50)
        if RATOON_EXCLUDE:   # bare Dec-Jan, regrowth by Mar-Apr (independent review 14 Oct: 9/24 missed cane, 5/250 non-cane)
            lo, hi = np.fmax(g("S2_NDVI_PREV12"), g("S2_NDVI_Y01")), np.fmax(g("S2_NDVI_Y03"), g("S2_NDVI_Y04"))
            m |= (lo < 0.30) & (hi >= 0.40) & (hi - g("S2_NDVI_Y01") >= 0.15)
        return m


REVIEW_ONLY = [s for s in os.environ.get("REVIEW_ONLY", "").split(",") if s]   # sites without a 2025 label: only their review / user label px are used (21 Oct 2026, late-sown cane)


def site_data(site, year, k):
    root, feat = paths(site, year)
    if site in REVIEW_ONLY:
        with rasterio.open(root / "data/grid/index.tif") as r:
            clean, edge = np.full(r.shape, 255, "uint8"), np.zeros(r.shape, "uint8")
    else:
        with rasterio.open(root / "data/labels/cane2025_qc.tif") as r:
            clean, edge = r.read(1), r.read(7)
    if NEWCANE_EXCLUDE and year == 2026 and site not in REVIEW_ONLY:
        nc = new_cane_mask(root, year)
        if nc is not None:
            log(f"{site}: {int(((clean == 0) & nc).sum()):,} non-cane px look like Oct-Dec plantings -> excluded")
            clean = np.where((clean == 0) & nc, 255, clean)
    PH = pheno_block(root, year) if (PHENO or PLANT_EXCLUDE) else None
    if PLANT_EXCLUDE and year == 2026 and PH is not None and site not in REVIEW_ONLY:
        g = lambda k: PH[0][PH[1].index("PH_" + k)].reshape(clean.shape)  # noqa: E731
        with np.errstate(invalid="ignore"):
            pc = (g("ndvi_dry_min") < 0.30) & (g("ndvi_mayjun_max") >= 0.55) & (g("ndvi_julsep_med") >= 0.60) & (g("greenup_days") <= 200)
        log(f"{site}: {int(((clean == 0) & pc).sum()):,} non-cane px look like 2026 plant cane -> excluded")
        clean = np.where((clean == 0) & pc, 255, clean)
    names, X = read_feat(feat, year)
    H, W = clean.shape
    Xf = X.reshape(len(names), -1)
    ok = (Xf > -32000).all(0)
    rng = np.random.default_rng(SEED + k)
    idx = []
    for cls in (1, 0):
        c = np.flatnonzero((clean.ravel() == cls) & (edge.ravel() >= 2) & ok)
        idx.append(rng.choice(c, min(N_PER_CLASS, len(c)), replace=False))
    idx = np.sort(np.concatenate(idx))
    F, cols = M.build(Xf[:, idx].T, names, FS)
    S2 = s2_block(root, year) if S2FEAT else None
    if S2 is not None:
        F = np.hstack([F, S2[0][:, idx].T]); cols = cols + S2[1]
    if PHENO:
        assert PH is not None, f"{site}: pheno_{year}.tif missing"
        F = np.hstack([F, PH[0][:, idx].T]); cols = cols + PH[1]
    elif PH is not None:
        PH = None                                        # mask only (PLANT_EXCLUDE without PHENO features)
    y, w = clean.ravel()[idx].astype(int), np.full(len(idx), SITE_W.get(site, 1.0), "float32")
    # blind-review answers of this district (2026 only) -> hard examples
    rev = [pd.read_csv(f) for f in sorted(root.glob("review*/review_results.csv"))] if year == 2026 else []
    if year == 2026:
        rev += [x[x.site == site] for x in (pd.read_csv(f) for f in EXTRA_REVIEW)]
    n_rev = 0
    for d in rev:
        d = d[d.answer.isin(["cane", "not_cane"]) & d.confidence.isin(REVIEW_CONF)]
        rr = np.concatenate([d.row.values + dr for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        cc = np.concatenate([d.col.values + dc for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        yy = np.tile((d.answer == "cane").astype(int).values, 9)
        keep = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        ii = rr[keep] * W + cc[keep]
        good = ok[ii]
        Fr, _ = M.build(Xf[:, ii[good]].T, names, FS)
        if S2 is not None:
            Fr = np.hstack([Fr, S2[0][:, ii[good]].T])
        if PH is not None:
            Fr = np.hstack([Fr, PH[0][:, ii[good]].T])
        F, y = np.vstack([F, Fr]), np.concatenate([y, yy[keep][good]])
        w = np.concatenate([w, np.full(int(good.sum()), REVIEW_WEIGHT * SITE_W.get(site, 1.0), "float32")])
        n_rev += int(good.sum())
    log(f"{site} {year}: {len(idx):,} px + {n_rev} review px, {len(cols)} features")
    return F, y, w, cols, n_rev


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [site_data(s, 2025, k) for k, s in enumerate(S25)] + [site_data(s, 2026, 100 + k) for k, s in enumerate(S26)]
    cols = parts[0][3]
    assert all(p[3] == cols for p in parts), "feature columns differ"
    F = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); w = np.concatenate([p[2] for p in parts])
    mdl = M.make("lgbm").fit(F, y, sample_weight=w)
    joblib.dump({"model": mdl, "feature_set": FS, "columns": cols, "threshold": float(os.environ.get("THR", "0.5")), "s2feat": S2FEAT, "pheno": PHENO,
                 "train_sites": [f"{s}:2025" for s in S25] + [f"{s}:2026" for s in S26]}, OUT / "model.joblib")
    (OUT / "train.json").write_text(json.dumps({"sites_2025": S25, "sites_2026": S26, "n": int(len(y)),
                                                "review_px": int(sum(p[4] for p in parts)), "review_weight": REVIEW_WEIGHT,
                                                "ratoon_exclude": RATOON_EXCLUDE, "plant_exclude": PLANT_EXCLUDE, "pheno": PHENO, "extra_review": [str(x) for x in EXTRA_REVIEW]}, indent=1))
    log("saved", OUT / "model.joblib")


if __name__ == "__main__":
    main()

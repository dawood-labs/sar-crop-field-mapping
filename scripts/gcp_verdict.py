"""Step 3 of the GCP EDA: check every point's label against what the satellite data shows.

Each label gets agronomic tests (expected behaviour in the study area, Sep 2025 - Oct 2026). NDVI tests use only
cloud-free observations (medians per period); S1 tests use the 3x3 window VH (dB). The Whittaker curve is NOT used
for verdicts, because with lambda 1 it swings wildly across the 3-4 month monsoon gap without clear images.

Important data fact found in the class summary: bare DRY soil in Feb-Apr also gives VH < -20 dB (sugarcane and cassava
after harvest), so "standing water" is only tested in the monsoon window Jun-Aug 2026.

Standing water = at least 2 Sentinel-1 acquisitions (either track, 3x3) with VH <= -20 dB in Jun-Aug 2026 (single
dips are speckle: they occur on trees too). For Rice, water is supporting evidence only: Local rice is mostly broadcast
/ dry-seeded and many paddies never show a long standing-water phase (14 of 52 rice points have 0-1 such dates).

Verdict: OK (all tests pass) | Doubtful (one test fails or the test is weak) | Likely wrong (two or more fail)
         | Unusable label (mixed / ambiguous label) | Not verifiable (one-off crop, too little to compare)
         | No label (Cant Access: a suggestion is given from the data)
Output: data/gcps/eda/gcp_verdicts.csv and gcp_verdicts.gpkg (points + all features + tests + verdict + reasons)
"""
from __future__ import annotations

from pathlib import Path
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
EDA = PROJ / "data/gcps/eda"
TREES = {"Rubber", "Eucalyptus", "Durian", "Cashew Nut"}
MIXED = {"CassavaRubber", "Rubber Cassava", "Rubber After Cassava"}


def v(x):
    return None if pd.isna(x) else float(x)


def tests(r):
    """Return {test_name: (passed: bool|None, explanation)} for the point's own label (None = no data)."""
    nw = int(r.n_wet_le20)
    vh_mean = np.nanmean([r.vh_RO164_mean, r.vh_RO099_mean])
    vh_rise = r.vh_RO164_SepOct26 - r.vh_RO164_MarApr26
    T = {}
    water = (nw >= 2, f"{nw} S1 dates with VH <= -20 dB in Jun-Aug (standing water if >= 2)")
    no_water = (nw <= 1, f"{nw} S1 dates with VH <= -20 dB in Jun-Aug (paddy-like standing water if >= 2)")
    crop = r.crop
    if crop == "Rice":
        T["harvest_end2025"] = (None if pd.isna(r.ndvi_SepOct25) or pd.isna(r.ndvi_NovDec25) else
                                bool(r.ndvi_SepOct25 >= 0.45 and min(r.ndvi_NovDec25, r.ndvi_JanFeb26) <= 0.45),
                                f"NDVI Sep-Oct25 {r.ndvi_SepOct25:.2f} -> Nov-Feb {min(r.ndvi_NovDec25, r.ndvi_JanFeb26):.2f} (green, then harvested)")
        T["dry_season_fallow"] = (None if pd.isna(r.ndvi_dry_max) else bool(r.ndvi_dry_max <= 0.5),
                                  f"max NDVI Jan-Apr {r.ndvi_dry_max:.2f} (no crop in dry season if <= 0.50)")
        T["standing_water"] = (bool(water[0]), water[1])
        T["crop_now"] = (bool((not pd.isna(r.ndvi_SepOct26) and r.ndvi_SepOct26 >= 0.45) or vh_rise >= 1.0),
                         f"NDVI Sep-Oct26 {r.ndvi_SepOct26:.2f}, VH rise Mar-Apr -> Sep-Oct {vh_rise:+.1f} dB (new crop)")
    elif crop in TREES:
        lo = 0.30 if crop == "Rubber" else 0.35            # rubber sheds its leaves (wintering) in Jan-Feb
        T["green_all_year"] = (None if pd.isna(r.ndvi_dry_min) else bool(r.ndvi_dry_min >= lo and r.ndvi_MarApr26 >= 0.45),
                               f"min NDVI Jan-Apr {r.ndvi_dry_min:.2f}, median Mar-Apr {r.ndvi_MarApr26:.2f} (canopy if >= {lo} / 0.45)")
        T["no_clearing"] = (bool(np.nanmin([r.ndvi_NovDec25, r.ndvi_MarApr26, r.ndvi_MayJun26]) >= 0.4),
                            f"lowest period NDVI {np.nanmin([r.ndvi_NovDec25, r.ndvi_MarApr26, r.ndvi_MayJun26]):.2f} (cut / cleared if < 0.40)")
        T["tree_backscatter"] = (bool(vh_mean >= -16.5), f"mean VH {vh_mean:.1f} dB (tree canopy if >= -16.5)")
        T["no_water"] = (bool(no_water[0]), no_water[1])
    elif crop == "Sugarcane":
        hi = np.nanmax([r.ndvi_SepOct25, r.ndvi_NovDec25])
        lo = np.nanmin([r.ndvi_JanFeb26, r.ndvi_MarApr26])
        T["tall_before_harvest"] = (bool(hi >= 0.55), f"max NDVI Sep-Dec25 {hi:.2f}")
        lo2 = np.nanmin([r.ndvi_NovDec25, r.ndvi_dry_min])              # single clear dates count: a cut field is bare briefly
        T["harvest_Nov_Apr"] = (bool(lo2 <= 0.35), f"lowest NDVI Nov-Apr {lo2:.2f} (cut if <= 0.35)")
        T["not_rice_pattern"] = (bool(r.ndvi_NovDec25 >= 0.4 or r.ndvi_MarApr26 >= 0.35 or r.ndvi_MayJun26 >= 0.4),
                                 f"NDVI Nov-Dec {r.ndvi_NovDec25:.2f}, Mar-Apr {r.ndvi_MarApr26:.2f}, May-Jun {r.ndvi_MayJun26:.2f} "
                                 "(cane is still standing in Nov-Dec or regrows in the dry season; bare Nov-Jun fits rice)")
        T["regrowth_now"] = (bool((r.ndvi_SepOct26 >= 0.5) if not pd.isna(r.ndvi_SepOct26) else r.vh_RO164_SepOct26 >= -16),
                             f"NDVI Sep-Oct26 {r.ndvi_SepOct26:.2f}, VH {r.vh_RO164_SepOct26:.1f} dB (ratoon/new cane)")
        T["no_water"] = (bool(no_water[0]), no_water[1])
    elif crop == "Cassava":
        T["not_tree"] = (bool(r.ndvi_dry_min <= 0.4), f"min NDVI Jan-Apr {r.ndvi_dry_min:.2f} (harvest / young crop phase)")
        T["no_water"] = (bool(no_water[0]), no_water[1])
        T["crop_cycle"] = (bool(np.nanmax([r.ndvi_SepOct25, r.ndvi_NovDec25, r.ndvi_SepOct26]) >= 0.45),
                           "green canopy in at least one season")
    elif crop == "Bared Land":
        T["bare_now"] = (bool((r.ndvi_SepOct26 <= 0.35) if not pd.isna(r.ndvi_SepOct26) else r.vh_RO164_SepOct26 <= -16),
                         f"NDVI Sep-Oct26 {r.ndvi_SepOct26:.2f}, VH {r.vh_RO164_SepOct26:.1f} dB")
        T["bare_since_May"] = (bool(r.ndvi_MayJun26 <= 0.35) if not pd.isna(r.ndvi_MayJun26) else None,
                               f"NDVI May-Jun26 {r.ndvi_MayJun26:.2f}")
        T["no_crop_monsoon"] = (bool(r.vh_RO164_JulAug26 <= -16), f"VH Jul-Aug {r.vh_RO164_JulAug26:.1f} dB (vegetation raises VH)")
    return T


def suggest(r):
    """What the data looks like, for unlabelled / doubtful points."""
    vh_mean = np.nanmean([r.vh_RO164_mean, r.vh_RO099_mean])
    if vh_mean >= -13 and np.nanmax([r.ndvi_JanFeb26, r.ndvi_MarApr26]) < 0.4:
        return "built-up / village (bright radar, little vegetation)"
    if r.ndvi_dry_min >= 0.35 and r.ndvi_MarApr26 >= 0.45 and vh_mean >= -16.5:
        return "tree crop / perennial (green all year)"
    if (r.ndvi_SepOct25 >= 0.5 and min(r.ndvi_NovDec25, r.ndvi_JanFeb26) <= 0.45 and r.ndvi_dry_max <= 0.5
            and (r.n_wet_le20 >= 2 or r.ndvi_SepOct26 >= 0.45)):
        return "rice-like (harvest Nov-Dec, bare dry season, wet-season crop)"
    if r.ndvi_dry_max > 0.5 and r.ndvi_dry_min <= 0.35:
        return "crop grown in the dry season (irrigated rice / vegetables / cane regrowth)"
    if np.nanmax([r.ndvi_JanFeb26, r.ndvi_MarApr26, r.ndvi_SepOct26]) < 0.3:
        return "bare / very sparse all season"
    return "unclear"


def main():
    import pickle
    f = pd.read_csv(EDA / "gcp_features.csv")
    S = pickle.load(open(EDA / "gcp_series.pkl", "rb"))
    n = {}
    for k, s in S.items():
        g = s["s1_3x3"]; w = g[(g.date >= "2026-06-01") & (g.date <= "2026-08-31")]
        n[k] = int((w.VH <= -20).sum())
    f["n_wet_le20"] = f.gcp_id.map(n)
    out = []
    for r in f.itertuples():
        T = tests(r)
        quality = []
        if r.het_ndvi > 0.08:
            quality.append(f"mixed surroundings (5x5 NDVI std {r.het_ndvi:.2f}, point near a field edge?)")
        if r.n_clear < 20:
            quality.append(f"only {r.n_clear} cloud-free S2 dates")
        res = [p for p, _ in T.values() if p is not None]
        failed = [f"{k}: {e}" for k, (p, e) in T.items() if p is False]
        if r.crop == "Cant Access":
            verdict, reasons = "No label", f"looks like: {suggest(r)}"
        elif r.crop in MIXED:
            verdict, reasons = "Unusable label", f"mixed label '{r.crop}'; data looks like: {suggest(r)}"
        elif r.crop == "Pumpkin":
            verdict, reasons = "Not verifiable", f"single short-cycle crop; data looks like: {suggest(r)}"
        elif not T:
            verdict, reasons = "Not verifiable", suggest(r)
        else:
            nf = len(failed)
            verdict = "OK" if nf == 0 else "Doubtful" if nf == 1 else "Likely wrong"
            if r.crop == "Rice" and nf == 1 and T["standing_water"][0] is False:
                verdict = "OK (no clear flooding)"             # water is supporting evidence only for rice
            if r.crop == "Cassava" and verdict == "OK":
                verdict = "OK (weak test)"
            reasons = "all tests pass" if nf == 0 else " | ".join(failed)
            if nf:
                reasons += f" || data looks like: {suggest(r)}"
        if quality:
            reasons += " || quality: " + "; ".join(quality)
        row = r._asdict(); row.pop("Index")
        row.update({f"test_{k}": ("pass" if p else "fail" if p is False else "n/a") for k, (p, e) in T.items()},
                   verdict=verdict, reasons=reasons, looks_like=suggest(r))
        out.append(row)
    d = pd.DataFrame(out)
    # visual review of every flagged point (data/gcps/eda/visual_review.csv) overrides the rule verdict
    vr = EDA / "visual_review.csv"
    if vr.exists():
        d = d.merge(pd.read_csv(vr), on="gcp_id", how="left")
    else:
        d["visual_verdict"] = d["visual_note"] = d["class_2026"] = None
    d["final_verdict"] = d.visual_verdict.fillna(d.verdict)
    auto = {"Rice": "Rice", "Rubber": "Tree", "Eucalyptus": "Tree", "Cashew Nut": "Tree", "Durian": "Tree",
            "Sugarcane": "Sugarcane", "Cassava": "Cassava", "Bared Land": "Bare"}
    ok = d.final_verdict.str.startswith("OK")
    d["class_2026"] = d.class_2026.fillna(d.crop.map(auto).where(ok))
    d.loc[d.crop.eq("Cant Access") & d.class_2026.isna(), "class_2026"] = "No label"
    d["class_2026"] = d.class_2026.fillna("Exclude")
    d["usable_for_training"] = ok & ~d.class_2026.str.endswith("?") & ~d.class_2026.isin(["Exclude", "No label"])
    d.to_csv(EDA / "gcp_verdicts.csv", index=False)
    g = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326)
    g.to_file(EDA / "gcp_verdicts.gpkg", layer="gcp_verdicts", driver="GPKG")
    print(pd.crosstab(d.crop, d.final_verdict, margins=True).to_string())
    print(d[d.usable_for_training].class_2026.value_counts().to_string())


if __name__ == "__main__":
    main()

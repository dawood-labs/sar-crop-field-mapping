"""Blind review of the static cassava model (layer 2): blocks inside the cane-like mask, strata = model class
(281 cassava candidate / 1 sugarcane), interior (5 x 5 uniform class), >= 300 m from the user's polygons and the test set.
  python scripts/cassava_review.py sample <vtag>   -> sites/cassava_review_<vtag>/ (sheets/K###.png, RUBRIC.md, key/, labels/)
  python scripts/cassava_review.py compare <vtag>  -> review_results.csv + precision per class
Sheets = indep_review.sheet (same look as every other review). Reviewers answer the crop type, never see the map or key.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
VTAG = sys.argv[2] if len(sys.argv) > 2 else "v2"
sys.argv = [sys.argv[0], sys.argv[1], "s2on_r2_oct_rn5_l2b"]
os.environ.setdefault("OUT_NAME", f"cassava_review_{VTAG}"); os.environ.setdefault("ID_PREFIX", "K")
import indep_review as IR  # noqa: E402
import numpy as np, pandas as pd, rasterio, geopandas as gpd  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
import site_paths as SP

SITES = {"site86_2026": "site86", "site131_2026": "Site131", "site113_2026": "Site113"}
N = {281: 60, 1: 40}
RUBRIC = IR.RUBRIC.replace("Answer one row per sheet: sample_id,answer,what,confidence,note",
    "EVERY block here is already known to be a dense upland crop in October (cane-like). Your job is the CROP TYPE.\n"
    "Answer one row per sheet: sample_id,answer,what,confidence,note\n"
    "- answer: sugarcane | cassava | other   (other = rice, maize, tree, grass ... say what in 'what')") \
    .replace("- answer: cane | not_cane | unsure\n", "")


def sample():
    for x in ("sheets", "key", "labels", "google"): (IR.OUT / x).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(IR.SEED); rows = []
    polys = pd.concat([gpd.read_file(IR.PROJ / "sites/user_labels/gcps_country_2026-10-15_analysis.gpkg").to_crs(32648)[["site", "geometry"]],
                       gpd.read_file(IR.PROJ / "sites/user_labels/cane_polygons_2026-10-15.gpkg").to_crs(32648)[["site", "geometry"]],
                       gpd.read_file(IR.PROJ / "sites/user_labels/cassava_noise_site86_2026-10-15.gpkg").to_crs(32648).assign(site="site86")[["site", "geometry"]]])
    tot = {}
    for site in SITES:
        with rasterio.open(SP.site_dir(site) / f"data/model/cassava_static_{VTAG}/class_2026.tif") as r: tot[site] = r.read(1)
    area = {c: sum(int((tot[s] == c).sum()) for s in SITES) for c in N}
    for site, key in SITES.items():
        from scipy import ndimage
        cls = tot[site]
        with rasterio.open(SP.site_dir(site) / "data/model/cassava_static_v2/class_2026.tif") as r: tr = r.transform
        k5 = np.ones((5, 5), bool); cv = cls.astype("int32")
        uni = (ndimage.minimum_filter(cv, footprint=k5) == ndimage.maximum_filter(cv, footprint=k5))
        far = np.ones(cls.shape, bool)
        pp = polys[polys.site == key]
        if len(pp):
            buf = rasterize([(g.buffer(300), 1) for g in pp.geometry], out_shape=cls.shape, transform=tr, dtype="uint8").astype(bool); far &= ~buf
        tk = pd.read_csv(IR.PROJ / "sites/testset_v1/key/key.csv"); tk = tk[tk.site == site]
        for r_, c_ in zip(tk.row, tk.col): far[max(r_ - 30, 0):r_ + 31, max(c_ - 30, 0):c_ + 31] = False
        for c, n_all in N.items():
            n = max(2, round(n_all * (cls == c).sum() / max(area[c], 1)))
            ok = uni & far & (cls == c); rr, cc = np.nonzero(ok)
            k = rng.choice(len(rr), min(n, len(rr)), replace=False)
            for r_, c_ in zip(rr[k], cc[k]):
                rows.append({"site": site, "stratum": "cassava_candidate" if c == 281 else "sugarcane", "row": int(r_), "col": int(c_), "pixel_id": int(r_ * cls.shape[1] + c_),
                             "label2025": -1, "map_tag": int(c), "n_stratum_px": int((cls == c).sum()), "n_eligible_px": len(rr), "weight": (cls == c).sum() / max(len(k), 1)})
        print(site, {c: int((cls == c).sum()) for c in N}, flush=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=IR.SEED).reset_index(drop=True)
    df["sample_id"] = [f"K{i + 1:03d}" for i in range(len(df))]
    trs = {s: rasterio.open(SP.site_dir(s) / "data/grid/index.tif").transform for s in SITES}
    x = [trs[r.site].c + (r.col + .5) * 10 for r in df.itertuples()]; y = [trs[r.site].f - (r.row + .5) * 10 for r in df.itertuples()]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=32648); ll = g.to_crs(4326).geometry
    df["lon"], df["lat"] = ll.x.round(6), ll.y.round(6); g["lon"], g["lat"] = df.lon, df.lat
    df.to_csv(IR.OUT / "key/key.csv", index=False); g.to_parquet(IR.OUT / "key/samples.parquet"); (IR.OUT / "RUBRIC.md").write_text(RUBRIC)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(r.site, r.sample_id, r.row, r.col, r.lon, r.lat) for r in df.itertuples()]
    with ProcessPoolExecutor(os.cpu_count() or 8, mp_context=mp.get_context("spawn")) as ex: list(ex.map(IR.sheet, jobs, chunksize=2))
    print(df.groupby(["site", "stratum"]).size().to_string(), f"\n{len(df)} sheets -> {IR.OUT / 'sheets'}")


def compare():
    key = pd.read_csv(IR.OUT / "key/key.csv")
    lab = pd.concat([pd.read_csv(f) for f in sorted((IR.OUT / "labels").glob("*.csv"))], ignore_index=True)
    d = key.merge(lab[["sample_id", "answer", "what", "confidence", "note"]], on="sample_id", how="left"); d.to_csv(IR.OUT / "review_results.csv", index=False)
    print(pd.crosstab(d.stratum, d.answer).to_string()); print()
    for st in ("cassava_candidate", "sugarcane"):
        x = d[d.stratum == st]; w = x.weight
        for ans in ("cassava", "sugarcane", "other"):
            print(f"{st:18s} reviewers say {ans:10s}: {(x.answer == ans).mean() * 100:5.1f} % of blocks, {(w * (x.answer == ans)).sum() / w.sum() * 100:5.1f} % of area")
    print(); print(pd.crosstab([d.site, d.stratum], d.answer).to_string())


if __name__ == "__main__":
    {"sample": sample, "compare": compare}[sys.argv[1]]()

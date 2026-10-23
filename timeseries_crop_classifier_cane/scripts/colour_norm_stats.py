"""Per-image colour normalisation statistics (21 Oct 2026, user: absolute colours differ by date and district; use RELATIVE colour).
For every site and every clear date >= 10 Sep: median and IQR of each colour feature over the clear pixels of the layer-1
cane-like mask (s2on_r2_oct_rn5). Training samples and apply pixels are standardised with the SAME stats: z = (x - median) / IQR.
Output: sites/layer2/colour_norm_stats.parquet (site, date, feature, median, iqr, n_px)
usage: python scripts/colour_norm_stats.py [site_dir ...]"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio
sys.path.insert(0, str(Path(__file__).resolve().parent)); os.environ.setdefault("COLOUR_TAG", "v8")
import colour_multi as C
import site_paths as SP
P = C.P; START = "20260910"; FCOLS = list(C.COLS) + ["B4", "B5/B4", "B4/B3"]


def feats11(b2, b3, b4, b5):
    X = pd.DataFrame(C.feats(b2, b3, b5), columns=C.COLS); X["B4"] = b4; X["B5/B4"] = b5 / (b4 + 1e-6); X["B4/B3"] = b4 / (b3 + 1e-6); return X


def one(sd, rng=np.random.default_rng(0)):
    S = SP.site_dir(sd) / "data"; st = S / "stacks"; rows = []
    with rasterio.open(S / "model/s2on_r2_oct_rn5/cane_map_2026.tif") as r: mask = r.read(1) == 1
    with rasterio.open(st / "S2_cs_cdf.tif") as c: dates = list(c.descriptions)
    src = {b: rasterio.open(st / f"S2_{b}.tif") for b in ("B2", "B3", "B4", "B5", "cs_cdf")}
    for i, d in enumerate(dates):
        if d < START: continue
        cs = src["cs_cdf"].read(i + 1); ok = mask & (cs >= 60) & (cs <= 100)
        if ok.sum() < 2000: continue
        idx = np.flatnonzero(ok.ravel()); idx = rng.choice(idx, min(len(idx), 200000), replace=False)
        b = {k: src[k].read(i + 1).ravel()[idx].astype("float32") / 1e4 for k in ("B2", "B3", "B4", "B5")}
        X = feats11(b["B2"], b["B3"], b["B4"], b["B5"])
        q = X.quantile([.25, .5, .75])
        for f in FCOLS: rows.append(dict(site=sd, date=d, feature=f, median=float(q.loc[.5, f]), iqr=float(q.loc[.75, f] - q.loc[.25, f]), n_px=int(ok.sum())))
    for s in src.values(): s.close()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sites = sys.argv[1:] or sorted(p.name for p in SP.all_sites("*_2026") if (p / "data/model/s2on_r2_oct_rn5").exists())
    out = P / "sites/layer2/colour_norm_stats.parquet"; old = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    new = pd.concat([one(s) for s in sites], ignore_index=True)
    allst = pd.concat([old[~old.site.isin(sites)] if len(old) else old, new], ignore_index=True); allst.to_parquet(out, index=False)
    print(new.groupby("site").date.nunique().to_dict(), "->", out)

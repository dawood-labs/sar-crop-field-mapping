"""Layer 2 (crop type inside the cane-like mask): features at every reviewed block + the test-set blocks (scoring only).

Three feature groups per 3 x 3 block (centre mean) with 5 x 5 texture:
  sar_*   Sentinel-1 late-season composite from the 15-day features (periods 202609a, 202609b, 202610a): per orbit mean and
          temporal std of VV, VH, VH-VV; 5 x 5 spatial std of VH and VV (texture); Aug (202608a/b) -> Sep-Oct change of VH.
  opt_*   Sentinel-2 "latest clear pixel" composite from 15 Sep 2026 (per pixel the most recent date with cs_cdf >= 60):
          B2 B3 B4 B5 B8, NDVI, NDRE (B8-B5)/(B8+B5), GNDVI, 5 x 5 std of NDVI and B8, days since 15 Sep, n clear dates.
  sea_*   season features from pheno_2026.tif (already computed for layer 1).
Output: sites/layer2/points.parquet (one row per block: site, row, col, what, answer, confidence, source, split, features)
usage: python scripts/layer2_points.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_resources  # noqa: F401,E402
import time, warnings  # noqa: E402
import numpy as np, pandas as pd, rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402
import site_paths as SP
warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parents[1]; OUT = PROJ / "sites/layer2"; OUT.mkdir(exist_ok=True)
T0 = time.time(); START = pd.Timestamp("2026-09-15")

lab = pd.read_csv(PROJ / "sites/layer2_labels_all.csv"); lab = lab[lab.site.str.endswith("_2026")].copy(); lab["split"] = "train"
key = pd.read_csv(PROJ / "sites/testset_v1/key/key.csv"); fin = pd.read_csv(PROJ / "sites/testset_v1/merge/final_labels.csv")
ts = key.merge(fin[["sample_id", "what_A", "what_B", "what_C", "label"]], on="sample_id")
def majority_what(r):
    w = [x for x in (r.what_A, r.what_B, r.what_C) if isinstance(x, str) and x not in ("unknown", "nan")]
    return max(set(w), key=w.count) if w else "unknown"
ts["what"] = ts.apply(majority_what, axis=1); ts["answer"] = ts.label.map({1: "cane", 0: "not_cane", -1: "unsure"}); ts["confidence"] = "n/a"; ts["source"] = "testset"; ts["split"] = "test"
pts = pd.concat([lab[["site", "row", "col", "answer", "what", "confidence", "source", "split"]], ts[["site", "row", "col", "answer", "what", "confidence", "source", "split"]]], ignore_index=True)
print(len(pts), "blocks", pts.split.value_counts().to_dict(), flush=True)

def win(r, c, k=5):
    return Window(c - k // 2, r - k // 2, k, k)

rows = []
for site, g in pts.groupby("site"):
    d = SP.site_dir(site) / "data"
    f1 = rasterio.open(d / "features/s1_15d_2026nov.tif"); n1 = list(f1.descriptions)
    idx = {(o, v, p): n1.index(f"{o}_{v}_{p}") for o in ("RO164_DES", "RO099_ASC") for v in ("VV", "VH", "VHmVV") for p in ("202608a", "202608b", "202609a", "202609b", "202610a")}
    ph = rasterio.open(d / "features/pheno_2026.tif"); nph = list(ph.descriptions)
    st = d / "stacks"; have_opt = all((st / f"S2_{b}.tif").exists() for b in ("B2", "B3", "B4", "B5", "B8", "cs_cdf"))
    if have_opt:
        s2 = {b: rasterio.open(st / f"S2_{b}.tif") for b in ("B2", "B3", "B4", "B5", "B8", "cs_cdf")}
        dates = pd.to_datetime(list(s2["cs_cdf"].descriptions)); late = np.flatnonzero(dates >= START)
        dstr = [x.strftime("%Y%m%d") for x in dates]
        sx_all = sorted((d / "raw/s2_extra").glob("S2X_*.tif")) if (d / "raw/s2_extra").exists() else []
        sx_files = [f for f in sx_all if f.stem[4:] in dstr]; sx_idx = [dstr.index(f.stem[4:]) for f in sx_files]
    else:
        sx_files = []
    for _, p in g.iterrows():
        r, c = int(p.row), int(p.col); o = p.to_dict()
        # ---- SAR composite
        a = f1.read(window=win(r, c)).astype("float32"); a[a <= -32000] = np.nan; a /= 100
        for orb, tag in (("RO164_DES", "des"), ("RO099_ASC", "asc")):
            for v in ("VV", "VH", "VHmVV"):
                so = np.stack([a[idx[(orb, v, pp)]] for pp in ("202609a", "202609b", "202610a")])   # 3 x 5 x 5
                au = np.stack([a[idx[(orb, v, pp)]] for pp in ("202608a", "202608b")])
                o[f"sar_{tag}_{v}_mean"] = np.nanmean(so[:, 1:4, 1:4]); o[f"sar_{tag}_{v}_tstd"] = np.nanmean(np.nanstd(so, 0)[1:4, 1:4])
                o[f"sar_{tag}_{v}_tex5"] = np.nanstd(np.nanmean(so, 0)); o[f"sar_{tag}_{v}_dAug"] = np.nanmean(so[:, 1:4, 1:4]) - np.nanmean(au[:, 1:4, 1:4])
        # ---- optical latest-clear composite
        if have_opt and len(late):
            cs = s2["cs_cdf"].read([int(i) + 1 for i in late], window=win(r, c)).astype("float32")
            b = {k: s2[k].read([int(i) + 1 for i in late], window=win(r, c)).astype("float32") for k in ("B2", "B3", "B4", "B5", "B8")}
            ok = (cs >= 60) & (cs <= 100) & (b["B8"] > 0)
            last = np.where(ok.any(0), ok.shape[0] - 1 - np.argmax(ok[::-1], 0), -1)      # latest clear index per pixel
            comp = {}
            for k in b:
                v = np.take_along_axis(b[k], np.maximum(last, 0)[None], 0)[0]; v[last < 0] = np.nan; comp[k] = v / 10000
            ndvi = (comp["B8"] - comp["B4"]) / (comp["B8"] + comp["B4"] + 1e-6); ndre = (comp["B8"] - comp["B5"]) / (comp["B8"] + comp["B5"] + 1e-6)
            gndvi = (comp["B8"] - comp["B3"]) / (comp["B8"] + comp["B3"] + 1e-6)
            for k in b: o[f"opt_{k}"] = np.nanmean(comp[k][1:4, 1:4])
            o["opt_ndvi"] = np.nanmean(ndvi[1:4, 1:4]); o["opt_ndre"] = np.nanmean(ndre[1:4, 1:4]); o["opt_gndvi"] = np.nanmean(gndvi[1:4, 1:4])
            o["opt_ndvi_tex5"] = np.nanstd(ndvi); o["opt_b8_tex5"] = np.nanstd(comp["B8"]); o["opt_ndvi_tex3"] = np.nanstd(ndvi[1:4, 1:4])
            lc = last[2, 2]; o["opt_days"] = (dates[late[lc]] - START).days if lc >= 0 else np.nan; o["opt_nclear"] = int(ok[:, 2, 2].sum())
        # ---- red-edge / SWIR latest-clear composite (data/raw/s2_extra, 15 Sep +; cloud mask from the cs_cdf stack)
        if have_opt and sx_files:
            csx = s2["cs_cdf"].read([i + 1 for i in sx_idx], window=win(r, c)).astype("float32")
            bx = np.stack([rasterio.open(f).read(window=win(r, c)).astype("float32") for f in sx_files])      # T x 5 x 5 x 5
            okx = (csx >= 60) & (csx <= 100) & (bx[:, 2] > 0)
            lastx = np.where(okx.any(0), okx.shape[0] - 1 - np.argmax(okx[::-1], 0), -1)
            cx = {}
            for k, nmx in enumerate(("B6", "B7", "B8A", "B11", "B12")):
                v = np.take_along_axis(bx[:, k], np.maximum(lastx, 0)[None], 0)[0]; v[lastx < 0] = np.nan; cx[nmx] = v / 10000
            ndre2 = (cx["B8A"] - cx["B6"]) / (cx["B8A"] + cx["B6"] + 1e-6); ndmi = (cx["B8A"] - cx["B11"]) / (cx["B8A"] + cx["B11"] + 1e-6)
            nbr = (cx["B8A"] - cx["B12"]) / (cx["B8A"] + cx["B12"] + 1e-6); sw_ratio = cx["B11"] / (cx["B8A"] + 1e-6)
            for k in cx: o[f"opx_{k}"] = np.nanmean(cx[k][1:4, 1:4])
            o["opx_ndre2"] = np.nanmean(ndre2[1:4, 1:4]); o["opx_ndmi"] = np.nanmean(ndmi[1:4, 1:4]); o["opx_nbr"] = np.nanmean(nbr[1:4, 1:4]); o["opx_swratio"] = np.nanmean(sw_ratio[1:4, 1:4])
            o["opx_ndmi_tex5"] = np.nanstd(ndmi); o["opx_b11_tex5"] = np.nanstd(cx["B11"]); o["opx_ndre2_tex5"] = np.nanstd(ndre2)
        # ---- season features (pheno)
        v = ph.read(window=win(r, c, 3)).astype("float32"); v[v <= -32000] = np.nan; v = np.nanmean(v, axis=(1, 2))
        for i, n in enumerate(nph):
            o[f"sea_{n}"] = v[i] if (n.split("_", 2)[-1] in ("last_cut", "grow_len", "grow_total") or n in ("greenup_days", "n_clear_mayoct")) else v[i] / 100
        rows.append(o)
    f1.close(); ph.close()
    if have_opt:
        for s in s2.values(): s.close()
    print(f"[{time.time() - T0:5.0f} s] {site}: {len(g)} blocks, optical {'yes' if have_opt else 'NO'}, red-edge dates {len(sx_files) if have_opt else 0}", flush=True)
df = pd.DataFrame(rows); df.to_parquet(OUT / "points.parquet", index=False)
print("saved", OUT / "points.parquet", df.shape)

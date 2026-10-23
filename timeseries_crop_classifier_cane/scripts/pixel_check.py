"""Quick check of pixel ids on a site's current map: python scripts/pixel_check.py <site> <model> id id ..."""
import sys, rasterio, numpy as np, warnings; warnings.filterwarnings("ignore")
import site_paths as SP
site, model = sys.argv[1], sys.argv[2]; ids=[int(x) for x in sys.argv[3:]]
S=f"{SP.site_dir(site)}"
with rasterio.open(f"{S}/data/grid/index.tif") as s: W=s.width; aoi=s.read(2)
rc=[(i//W,i%W) for i in ids]; R=[r for r,c in rc]; C=[c for r,c in rc]
def px(p,b=1):
    with rasterio.open(p) as s: return s.read(b)[R,C]
d=f"{S}/data/model/{model}"
with rasterio.open(f"{S}/data/stacks/S2_B4.tif") as s: dd=list(s.descriptions)
with rasterio.open(f"{S}/data/stacks/S2_B4.tif") as r4, rasterio.open(f"{S}/data/stacks/S2_B8.tif") as r8, rasterio.open(f"{S}/data/stacks/S2_cs.tif") as rc_:
    n=[]; 
    for b in range(len(dd)):
        b4=r4.read(b+1)[R,C].astype(float); b8=r8.read(b+1)[R,C].astype(float); cs=rc_.read(b+1)[R,C]
        v=(b8-b4)/(b8+b4+1e-6); v[cs<60]=np.nan; n.append(v)
n=np.array(n); dd=np.array(dd)
def stat(lo,hi,f): 
    m=(dd>=lo)&(dd<=hi); return f(n[m],0) if m.any() else np.full(len(ids),np.nan)
out=dict(in_aoi=aoi[R,C], prob=px(f"{d}/cane_prob_2026.tif"), map=px(f"{d}/cane_map_2026.tif"), oct_min=px(f"{d}/oct_ndvi_min.tif"),
 label25=px(f"{S}/data/labels/cane2025_labels.tif"),
 ndvi_dec_feb_min=stat("20251201","20260228",np.nanmin), ndvi_mar_apr_max=stat("20260301","20260430",np.nanmax),
 ndvi_may_jun_max=stat("20260501","20260630",np.nanmax), ndvi_jul_oct_med=stat("20260701","20261031",np.nanmedian),
 n_clear_jul_oct=((dd>="20260701")&(dd<="20261031")&~np.isnan(n).all(1)).sum() if False else np.sum(~np.isnan(n[(dd>="20260701")]),0))
import pandas as pd; pd.set_option("display.width",200)
print(pd.DataFrame(out,index=ids).round(2).to_string())

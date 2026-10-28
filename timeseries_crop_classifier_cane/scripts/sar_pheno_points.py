"""Date-free SAR phenology metrics at the test-set blocks (3x3 mean) and simple cane-like rules.
Usage: python scripts/sar_pheno_points.py   -> sites/testset_v1/sar_pheno_rows.csv + rule scores
Metrics per orbit from the 15-day VH composites (Nov 2025 - Oct 2026), then merged (mean / extreme):
  last_cut     : index (composite #) of the last deep dip (VH <= -19 dB) before the season end
  grow_len     : number of consecutive composites with VH >= -16.5 ending at the last composite (canopy persistence)
  grow_total   : number of composites with VH >= -16.5 after the last cut
  peak_vh      : max VH after the last cut
  end_vh       : median VH of the last 4 composites (Aug b - Oct a)
  flood        : min VH in the wet half (composite >= Jun a) (<= -21 = flooded paddy)
  dip_depth    : min VH over the year
  year_std     : std of VH over the year (forest / orchard ~ small)
"""
import pandas as pd, rasterio, numpy as np, warnings; warnings.filterwarnings("ignore")
from rasterio.windows import Window
TS="sites/testset_v1"; ROOT="sites"
k=pd.read_csv(f"{TS}/score_s2on_r2_oct_rows.csv").set_index("sample_id"); k=k[k.label>=0]
with rasterio.open(f"{ROOT}/site86_2026/data/features/s1_15d_2026nov.tif") as s: names=np.array(s.descriptions)
def metrics(vh, slopes=None):
    vh=np.asarray(vh, float); n=len(vh); ok=~np.isnan(vh)
    if ok.sum()<10: return dict(last_cut=np.nan,grow_len=np.nan,grow_total=np.nan,peak_vh=np.nan,end_vh=np.nan,flood=np.nan,dip=np.nan,ystd=np.nan)
    v=pd.Series(vh).interpolate(limit_direction="both").values
    dips=np.where(v<=-19)[0]; last_cut=int(dips[-1]) if len(dips) else -1
    g=0
    for x in v[::-1]:
        if x>=-16.5: g+=1
        else: break
    after=v[last_cut+1:] if last_cut>=0 else v
    return dict(last_cut=last_cut, grow_len=g, grow_total=int((after>=-16.5).sum()), peak_vh=after.max() if len(after) else np.nan,
                end_vh=np.median(v[-4:]), flood=v[n//2:].min(), dip=v.min(), ystd=v.std())
rows=[]
for site,g in k.groupby("site"):
    with rasterio.open(f"{ROOT}/{site}/data/features/s1_15d_2026nov.tif") as s:
        nm=np.array(s.descriptions); nd=s.nodata
        for sid,r in g.iterrows():
            a=s.read(window=Window(r.col-1,r.row-1,3,3)).astype(float); a[a==nd]=np.nan
            v=np.nanmean(a,axis=(1,2));  v=v/100 if np.nanmax(np.abs(v))>100 else v
            d={"sample_id":sid}
            for orb in ("RO164_DES","RO099_ASC"):
                idx=[i for i,x in enumerate(nm) if x.startswith(f"{orb}_VH_")]
                for kk,val in metrics(v[idx]).items(): d[f"{orb}_{kk}"]=val
            rows.append(d)
f=pd.DataFrame(rows).set_index("sample_id")
for kk in ("last_cut","grow_len","grow_total","peak_vh","end_vh","flood","dip","ystd"):
    a,b=f[f"RO164_DES_{kk}"],f[f"RO099_ASC_{kk}"]
    f[kk]= pd.concat([a,b],axis=1).min(1) if kk in ("flood","dip") else pd.concat([a,b],axis=1).mean(1)
k=k.join(f); k.to_csv(f"{TS}/sar_pheno_rows.csv")
m=pd.read_csv(f"{TS}/merge/final_labels.csv").set_index("sample_id"); k["what"]=m.what_A
pd.set_option("display.width",220)
print(k.groupby("label")[["last_cut","grow_len","grow_total","peak_vh","end_vh","flood","dip","ystd"]].describe().loc[:, (slice(None), ["25%","50%","75%"])].round(1).T.to_string())
w=k.weight; y=k.label
def pr(p,name):
    tp=(w*((y==1)&p)).sum(); fp=(w*((y==0)&p)).sum(); fn=(w*((y==1)&~p)).sum()
    print(f"{name:60s} P {tp/(tp+fp):.3f} R {tp/(tp+fn):.3f} F1 {2*tp/(2*tp+fp+fn):.3f} area {((w*p).sum()*0.0247105/1000):.0f}k  FPblk {int(((y==0)&p).sum())} FNblk {int(((y==1)&~p).sum())}")
    return p
pr(k.pred==1,"current map s2on_r2_oct")
for gl in (4,6,8):
    for ev in (-16.5,-16,-15.5):
        pr((k.grow_len>=gl)&(k.end_vh>=ev)&(k.flood>-21)&(k.ystd>=1.5), f"grow_len>={gl} end_vh>={ev} noflood std>=1.5")
best=pr((k.grow_len>=6)&(k.end_vh>=-16)&(k.flood>-21)&(k.ystd>=1.5)&(k.dip<=-18), "best + harvest dip<=-18")
print("FP what:", k[(y==0)&best].what.value_counts().to_dict()); print("FN what:", k[(y==1)&~best].what.value_counts().to_dict())
print(k[(y==1)&~best][["site","what","grow_len","end_vh","flood","dip","ystd"]].round(1).to_string())

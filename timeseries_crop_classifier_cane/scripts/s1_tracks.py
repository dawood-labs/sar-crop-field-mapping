"""List Sentinel-1 tracks covering an AOI (coverage fraction, dates) for a date range.
usage: python scripts/s1_tracks.py <aoi.shp> [start] [end]"""
import sys, os, geopandas as gpd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sar_pipeline.auth import init_ee
import ee
CFG = {"_root": str(__import__("pathlib").Path(__file__).resolve().parents[1]), "auth": {"key_file": __import__("os").environ.get("EE_KEY_FILE", "secrets/ee_service_account.json"), "project": __import__("os").environ.get("EE_PROJECT", "")}}
init_ee(CFG)
aoi = gpd.read_file(sys.argv[1]).to_crs(4326); start = sys.argv[2] if len(sys.argv) > 2 else "2025-11-01"; end = sys.argv[3] if len(sys.argv) > 3 else "2026-10-15"
g = ee.Geometry(aoi.geometry.unary_union.__geo_interface__)
col = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(g).filterDate(start, end).filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
info = col.aggregate_array("system:index").getInfo(); ro = col.aggregate_array("relativeOrbitNumber_start").getInfo(); ps = col.aggregate_array("orbitProperties_pass").getInfo()
tracks = sorted({(r, p) for r, p in zip(ro, ps)})
area = g.area(1)
for r, p in tracks:
    c = col.filter(ee.Filter.eq("relativeOrbitNumber_start", r)).filter(ee.Filter.eq("orbitProperties_pass", p))
    cov = c.geometry().intersection(g, 1).area(1).divide(area).getInfo()
    dates = sorted({i.split("_")[4][:8] for i, rr, pp in zip(info, ro, ps) if rr == r and pp == p})
    print(f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[1]))))} RO{r:03d} {p[:3]} coverage {cov:.2f} dates {len(dates)} first {dates[0]} last {dates[-1]}", flush=True)

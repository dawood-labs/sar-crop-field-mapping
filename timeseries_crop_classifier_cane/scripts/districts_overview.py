"""One GeoParquet with every district AOI and its role (training / test only / not used), to view in QGIS.
usage: python scripts/districts_overview.py -> sites/districts_overview.parquet (+ .qml, colour by role)
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
D26 = ["site112", "site136", "site20", "site134", "site84", "site82", "site06",
       "site122", "site86"]
ROWS = [("site119", "data/aoi/aoi_site119.shp", "train", "2025 + 2026", "first AOI; final 2026 map")] + \
       [(s, f"sites/{s}/data/aoi/aoi.shp", "train", "2025", "2025 training site") for s in ("site21", "site01")] + \
       [(s, f"sites/{s}_2026/data/aoi/aoi.shp", "train", "2026", "2026 district loop (label QC + blind reviews)") for s in D26] + \
       [(s, f"sites/bench_{s}_2026/data/aoi/aoi.shp", "test_only", "2026", "fixed benchmark, never trained on") for s in ("site102", "site26")] + \
       [("site104", "sites/site104/data/aoi/aoi.shp", "not_used", "2025", "old external test"),
        ("site04", "sites/site04/data/aoi/aoi.shp", "not_used", "2025", "rice site, not used")]
COLORS = {"train": "#33ef2a", "test_only": "#F44336", "not_used": "#A0A0A0"}

parts = []
for name, f, role, years, note in ROWS:
    g = gpd.read_file(PROJ / f).to_crs(32648)
    parts.append(gpd.GeoDataFrame({"district": [name], "role": [role], "years": [years], "note": [note]},
                                  geometry=[g.union_all()], crs=32648))
out = pd.concat(parts, ignore_index=True)
out["area_ac"] = (out.area / 4046.8564224).round(3)
dst = PROJ / "sites/districts_overview.parquet"
out.to_parquet(dst)
cats = "".join(f'<category render="true" symbol="{i}" value="{k}" label="{k}"/>' for i, k in enumerate(COLORS))
syms = "".join(f'<symbol type="fill" name="{i}" alpha="1"><layer class="SimpleFill"><prop k="color" v="{c}"/>'
               f'<prop k="style" v="b_diagonal"/><prop k="outline_color" v="{c}"/><prop k="outline_width" v="0.8"/></layer></symbol>'
               for i, c in enumerate(COLORS.values()))
dst.with_suffix(".qml").write_text(f'<qgis><renderer-v2 type="categorizedSymbol" attr="role"><categories>{cats}</categories>'
                                   f'<symbols>{syms}</symbols></renderer-v2><labeling type="simple"><settings>'
                                   f'<text-style fieldName="district" fontSize="10"/></settings></labeling></qgis>')
print(out.drop(columns="geometry").to_string(index=False))

# Time-series crop classifier: standing sugarcane at province scale

Maps **standing sugarcane** (cane that is still in the field at the current date) for whole provinces at 10 m, from a
season of **Sentinel-1 radar** plus **Sentinel-2 optical** images, and delivers clean field polygons with acres.

The product has two layers:

| Layer | What it answers | How |
|---|---|---|
| 1. cane-like map | "is there a tall cane-like crop here now?" (sugarcane + look-alikes such as cassava) | LightGBM on the season's S1 15-day composites + S2 dry-season and phenology features, then two rules |
| 2. colour model | "inside layer 1, which part is really sugarcane?" | a small colour model on the latest clear S2 image (5-3-2 false colour: B5 / B3 / B2) |

Areas are always in **acres** (1 pixel = 100 m² = 0.0247 acre), computed in the metric UTM grid.

---

## 1. The idea in steps

1. **Download** (Google Earth Engine, direct `computePixels` reads onto a fixed 10 m grid): every Sentinel-1 VV / VH pass of one
   descending and one ascending orbit (Nov - Oct), and every Sentinel-2 date (B2 B3 B4 B5 B8 + Cloud Score+ and cloud masks)
   with less than 75 % cloud over the area.
2. **Stacks**: one multi-band GeoTIFF per variable, all dates of the season (`build_stacks.py`).
3. **Features**:
   - S1 15-day median composites of VV, VH and VH-VV per orbit (`s1_composites.py`, season `2026nov`);
   - S2 monthly NDVI / NDRE in the dry season (`s2_dry_features.py`);
   - phenology metrics: last cut, regrowth length, peak / end VH, green-up day, dry-season minimum NDVI (`pheno_features.py`).
4. **Layer 1** (`apply_site.py`): the trained LightGBM model gives a cane probability per pixel (threshold 0.5), then
   - **October bare rule** (`oct_bare_rule.py`): a pixel bare in any clear image from mid September is not standing cane;
   - **recall rule v5** (`recall_rule.py`): adds young plant cane the model misses (bare in the dry season, dense from July,
     stable radar texture, so no forest).
5. **Layer 2** (`colour_multi.py apply`, colour model v8): cane vs cassava from the colour of the latest clear image,
   smoothed with a 9 x 9 mean.
6. **Clean** (`sieve_layer2.py`): fill small holes that are fully enclosed by cane and drop isolated cane specks (< 20 px);
   field edges never move.
7. **Province products**:
   - `province_mosaic.py`: one province raster of layer 1 (values as they are);
   - `province_cane.py`: one province GeoPackage of sugarcane polygons (`area_ac`, 3 decimals; < 0.15 ac dropped),
     never cut at district lines;
   - single-date S2 images of the province for review (`download.py` on the province AOI).

Model training and evaluation scripts (`model_loop.py`, `testset_score.py`, review tools) are included. The test set is a set
of blind-reviewed blocks that is never used for training.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt          # numpy, pandas, geopandas, rasterio, lightgbm, earthengine-api, ...
export EE_PROJECT=<your earth engine project>
export EE_KEY_FILE=secrets/ee_service_account.json   # service account key, relative to this folder
export S3_BUCKET=<bucket> S3_PREFIX=<prefix>         # where raw data, stacks and outputs are backed up (optional)
```

Every script reads its inputs from a **site folder**: `sites/<year>/<Province>/<site>_<year>/` with `data/aoi/aoi.shp`.
`scripts/site_paths.py` finds the folder from a site name (`python scripts/site_paths.py dir <site>`).
CPU and memory use adapt to the machine (`scripts/safe_resources.py`): every core, workers sized to the guaranteed RAM.

---

## 3. Run one district

```bash
S=$(python scripts/site_paths.py dir site01_2026)
export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_START=2025-11-01 DL_END=2026-10-15
export DL_S1_TRACKS="164:DESCENDING,99:ASCENDING" S1_TRACKS="RO164_DES,RO099_ASC"
python scripts/download.py --only s1 && python scripts/download.py --only s2     # raw per-date files
python scripts/build_stacks.py                                                    # data/stacks/
python scripts/s1_composites.py 2026nov                                           # data/features/s1_15d_2026nov.tif
python scripts/s2_dry_features.py 2026 $S && python scripts/pheno_features.py 2026 $S
MODEL=data/model/s2on_r2/model.joblib OUT_TAG=s2on_r2 FEATURES_FILE=s1_15d_2026nov.tif YEAR=2026 python scripts/apply_site.py $S
python scripts/oct_bare_rule.py s2on_r2 s2on_r2_oct $S
RULE=5 python scripts/recall_rule.py s2on_r2_oct s2on_r2_oct_rn5 $S              # layer 1: data/model/s2on_r2_oct_rn5/
COLOUR_TAG=v8 OUT_TAG=layer2_final SMOOTH=9 LAYER1=s2on_r2_oct_rn5 python scripts/colour_multi.py apply site01_2026
```

Other orbits than the model's (RO164 / RO099) are renamed with `S1_TRACK_ALIAS="RO026_ASC:RO099_ASC"`.

## 4. Run a whole province

`drivers/province_run.sh` runs all districts of a province with one download lane and two processing lanes, uploads raw data,
stacks and outputs to S3 (size-verified) and then frees the local disk. Markers in each site folder (`DOWNLOADED`, `PROCESSING`,
`UPLOADED`) make a re-run skip finished work. `drivers/gap_run.sh` processes the parts of a province that no district polygon
covers. Then:

```bash
SIZE=20 python scripts/sieve_layer2.py apply <site> ...                  # per district clean (layer2_final_sieve/)
python scripts/province_mosaic.py <Province> data/model/s2on_r2_oct_rn5/cane_map_2026.tif <prov>_cane_map_2026.tif
python scripts/province_cane.py <Province>                               # <prov>_sugarcane_2026.gpkg
```

## 5. Outputs (per site, `data/model/`)

| Folder / file | Meaning |
|---|---|
| `s2on_r2_oct_rn5/cane_map_2026.tif` | layer 1: 1 cane-like, 0 not, 255 outside |
| `s2on_r2_oct_rn5/cane_source_2026.tif` | 1 = model, 2 = recall rule |
| `layer2_final/class3_2026.tif` | layer 2: 1 sugarcane, 281 cassava, 0 none, 65535 outside / cloud |
| `layer2_final_sieve/class_2026.tif` | layer 2 after the hole / speck clean |

Every raster has a QGIS style (`.qml`) next to it.

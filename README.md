# SAR crop field mapping

Field-level maps of **rice**, **sugarcane** and **cassava** for one district-sized area of interest (AOI), built
mainly from **Sentinel-1 radar** (works through monsoon clouds) and checked / corrected with **Sentinel-2 optical**
images where the sky is clear. The final product is one polygon per field with a class code, delivered as a
shapefile (client) and GeoParquet (internal).

Everything runs on one 10 m pixel grid. Areas are reported in **acres** (1 pixel = 100 m² = 0.0247 acre).

---

## 1. The idea in five steps

1. **Download** Sentinel-1 (every pass of the two orbits that fully cover the AOI) and Sentinel-2 (every date, with
   all cloud masks) straight from Google Earth Engine onto the fixed grid.
2. **Pixel features from radar**: monthly VV / VH / VH−VV backscatter, plus a few crop-calendar features
   (number of flooded dates in the monsoon, lowest VH, canopy growth, dry-season regrowth, late cut).
3. **Pixel classifiers** (XGBoost): rice vs non-rice, then cane-cassava vs the rest. Training labels come from
   ground points and from **blind reviews**: analysts label pixels on evidence sheets without seeing any model output;
   each review round becomes new training data and an honest test of the previous model.
4. **Fields**: the pixel map is cleaned (sieve 0.2 acre) and summarised per field polygon (majority rule); big
   fields that are clearly mixed are cut with straight lines; geometry is cleaned so no two polygons overlap or touch.
5. **Sentinel-2 QC**: field-level NDVI in five seasonal windows corrects confident disagreements, recovers missed
   cane fields and splits cane from cassava. A final QC checks every geometry before delivery.

---

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables (nothing secret is stored in the code):

| Variable | Meaning |
|---|---|
| `EE_PROJECT` | Google Cloud project used for Earth Engine |
| `EE_KEY_FILE` | service-account JSON key, path relative to the repo root (default `secrets/ee_service_account.json`) |
| `GCS_BUCKET`, `GCS_PREFIX` | only for `upload.py` (backup of `data/` to Cloud Storage) |
| `EE_HELPERS_SRC` | not needed: the helpers are bundled in `scripts/sar_pipeline/` |

Folders the scripts expect (created by the scripts unless noted; `data/` and `logs/` are never committed):

```
data/aoi/aoi.shp                        AOI polygon (you provide)
data/gcps/ground_samples/ground_samples.shp   ground points with a `Crop` column (you provide)
data/fields/delineation/fields_raw.gpkg field delineation polygons, layer "fields" (you provide)
data/grid/  data/raw/  data/stacks/  data/features/  data/model/  data/qc/  data/deliverables/
logs/
```

**Memory**: the scripts size their worker pools from the container's memory *request*
(`scripts/safe_resources.py`), not the free RAM, so a Kubernetes pod is not evicted. Run long jobs with
`nohup ... &` and `scripts/keep_alive_job.sh <PID>` (keeps a JupyterHub server from being culled);
`scripts/memwatch.sh` logs pod memory every 15 s.

---

## 3. Run order

Each step reads the outputs of the steps before it. Times are for an 8-core machine and a ~33,000 ha AOI.

| # | Command | Input | Output |
|---|---|---|---|
| 1 | `python scripts/download.py` (then `--only s2x`) | AOI, Earth Engine | `data/grid/`, `data/raw/s1`, `data/raw/s2*` |
| 2 | `python scripts/dem.py`, `python scripts/rain.py` | AOI | `data/ancillary/dem.tif`, `data/rain/` |
| 3 | `python scripts/build_stacks.py` | raw per-date files | `data/stacks/<variable>.tif` (one band per date) |
| 4 | `python scripts/precompute.py` | stacks | NDVI, clear-sky mask, smoothed NDVI, rain layers |
| 5 | `python scripts/gcp_extract.py`, `gcp_verdict.py`, `gcp_plots.py` | ground points | checked labels `data/gcps/eda/` |
| 6 | `python scripts/s1_features_v4.py`, `python scripts/s1_features_v6.py` | S1 stacks, rain | `data/features/*.tif` |
| 7 | `python scripts/rice_model_v5.py` | features + all labels | `data/model/rice_v5/` (rice probability + map) |
| 8 | `python scripts/cane_model.py cane_v3` | features + labels + rice map | `data/model/cane_v3/` |
| 9 | `python scripts/simplify_fields.py` | raw field delineation | clean polygons (0.5 m simplified, nested merged, 0.1 m gaps) |
| 10 | `python scripts/field_map_v5.py rice_v5` and `... cane_v3 Cane-Cassava cane_map.tif` | pixel maps + fields | field maps per class |
| 11 | `python scripts/clip_fields.py ...` (rice minus cane, then cane minus rice with `0.1`) | field maps | non-overlapping class layers |
| 12 | `python scripts/deliver_fields.py` | class layers | deliverable v1 + QC report |
| 13 | `python scripts/qc_s2_fields.py`, then `python scripts/qc_s2_v3.py` | deliverable + S2 | final deliverable (v3) |

Blind reviews (between model versions):

```bash
python scripts/audit_fields.py              # rice: 400 field-interior pixels, evidence sheets
python scripts/audit_cane.py sample cane_v1 # cane: 100 + 100 fields; analysts write labels/batch_*.csv
python scripts/audit_cane.py compare cane_v1
```

The analysts read `RUBRIC.md` and the PNG sheets only (S1 / S2 curves, image chips, terrain); the key file with the
model's answer is kept apart.

---

## 4. Scripts

| Area | Script | What it does |
|---|---|---|
| data | `download.py` | grid + S1 / S2 per date via `ee.data.computePixels`, resumable |
| | `build_stacks.py`, `precompute.py` | per-variable time stacks; NDVI, clear mask (Cloud Score+, ESA fallback), Whittaker NDVI |
| | `dem.py`, `rain.py`, `upload.py` | terrain, rainfall (IMERG / GSMaP / ERA5-Land / CHIRPS), backup to GCS |
| | `explorer.py`, `pixel_tools.py`, `notebooks/pixel_explorer.ipynb` | look at any pixel: curves, rain, chips |
| labels | `gcp_*.py`, `candidates*.py`, `labels_merge.py` | ground-point checks, candidate labels |
| | `audit_sheets.py`, `audit_fields.py`, `audit_compare.py`, `audit_cane.py` | blind evidence sheets, sampling, scoring |
| features | `s1_features.py`, `s1_features_v4.py`, `s1_features_v6.py` | radar features (window means in linear power, rainy dates dropped) |
| models | `rice_model*.py` (v1 – v6), `cane_model.py` | XGBoost, grouped spatial CV, honest comparison with the previous version |
| fields | `topology.py` | simplify, merge nested, refine (slits / holes / tails), separate (no overlap, no touching), checks |
| | `simplify_fields.py`, `field_map_v5.py`, `clip_fields.py`, `ftw_fields.py` | field layers from pixel maps; FTW = optional open field-boundary model |
| delivery | `deliver_fields.py`, `qc_s2_fields.py`, `qc_s2_v3.py` | deliverable + geometry QC, Sentinel-2 correction, cane / cassava split |
| infra | `safe_resources.py`, `keep_alive_job.sh`, `memwatch.sh`, `sar_pipeline/` | memory-safe workers, keep-alive, auth helpers |

Every script starts with a docstring that explains *why* it exists, its inputs, outputs and the exact command.

---

## 5. Output format

`data/deliverables/rice_cane_fields.zip` (shapefile) — one feature per field:

| Field | Meaning |
|---|---|
| `field_id` | running id |
| `predicted` | class code: 7 rice, 1 sugarcane, 281 cassava |
| `class` | class name |
| `area_ac` | area in acres |
| `conf` | sugarcane / cassava confidence (`high` = S2 probability ≥ 0.8 or ≤ 0.2) |

The GeoParquet copy carries every intermediate attribute (S2 probabilities, source of each polygon, earlier ids).
Guarantees checked before writing: valid single polygons, no overlap, no touching (a later dissolve never merges two
fields), no pin holes, everything inside the AOI.

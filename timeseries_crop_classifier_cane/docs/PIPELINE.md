# Standing-cane pipeline - run guide and requirements (for the Airflow DAG)

One command per task, one required input: the province. Everything else has a default, and all state is on S3.
A killed pod (spot reclaim) loses at most the stage that was running; the next run skips everything S3 already holds.

```
PYTHONPATH=scripts python -m pipeline run --province province5                     # whole province on one pod
PYTHONPATH=scripts python -m pipeline run --province province5 --units grid --fd true
```

## 1. What one province run does

| step | task | input | output on S3 (`<s3_root>/<year>/<Province>/`) |
|---|---|---|---|
| 1 | `prepare` | province name | `_plan/plan.json` + one AOI per unit under `districts/<unit>/data/aoi/` |
| 2 | `site` (one per unit, parallel) | plan + AOI | `districts/<unit>/data/{raw,stacks,features,model,grid,manifest}` |
| 3 | `province` | all units done | `province/<prov>_sugarcane_2026.gpkg`, `<prov>_class_2026_s20.tif`, `<prov>_cane_map_2026.tif`, meta |
| 4 | `fd` (one per version, optional) | province class raster + delineation fields | `fd_review/<prov>_fd_sugarcane_2026_<v>.gpkg` |

Units: `districts` = GAUL districts + gap AOIs (land of the province that no district covers); `grid` = square tiles of
`grid_km` on a fixed UTM grid clipped to the province (no gaps, equal-sized tasks; slivers merged into a neighbour).

Inside `site`: download (Sentinel-1 VV/VH on 2 tracks, Sentinel-2 B2 B3 B4 B5 B8 + cloud score, Earth Engine direct read)
-> stacks -> 15-day S1 composites -> S2 dry-season features -> phenology -> layer 1 (LightGBM `s2on_r2` + October bare rule +
recall rule v5) -> layer 2 (colour model v8) -> upload. The science code is unchanged from the locked production runs.

## 2. Parameters

Priority: CLI flag > env `PIPELINE_<NAME>` > `config/standing_cane_2026.json`.

| parameter | default | meaning |
|---|---|---|
| `province` | - (required) | e.g. `province5` (`_` for spaces; names of `provinces.parquet`) |
| `units` | `districts` | `districts` or `grid` |
| `grid_km` | 20 | grid tile size (20 km ~ 100k acres) |
| `min_tile_acres` | 5000 | grid slivers below this are merged into a neighbour tile |
| `fd` | false | also run the field-delineation polygons |
| `fd_versions` | `v1,1ac,1p5ac,2ac` | which FD versions |
| `year` / `season_start` / `season_end` | 2026 / 2025-11-01 / 2026-10-15 | season window of the download |
| `s3_root` | `s3://$S3_BUCKET/$S3_PREFIX` | where outputs + state are written |
| `inputs_root` | = `s3_root` (JSON: production root) | where the models, source tables and delineation inputs are read |
| `layer1_model` / `colour_model` | locked models on S3 | S3 URIs of the two models |
| `delineation_root` | `<s3_root>/shared/delineation` | holds `<Province>/*_fields_2026.parquet` + `*Boundary.gpkg` |
| `force` | "" | redo stages although done: `prepare,download,process,province,fd` |
| `keep_local` | false | keep raw + stacks on disk after upload |
| `min_free_gb` / `disk_wait_s` | 35 / 0 | disk needed before a download; wait instead of failing |
| `max_cpus` / `max_mem_gb` | 0 / 0 | caps (0 = whatever the pod gives; cgroup aware) |
| `ee_key_file` / `ee_project` | env `EE_KEY_FILE` / `ee-company` | Earth Engine service account (path or `s3://` URI) |
| `workload_tag` | `standing-cane` | Earth Engine billing label |
| `dry_run` | false | `run` / `prepare`: only show the plan |

## 3. DAG shape

```
prepare(province) -> plan(province) -> [site(unit) for unit in plan.sites]  (dynamic mapping, any parallelism)
                                    -> province(province)
                                    -> [fd(province, v) for v in fd_versions]   (only when fd = true)
```
`plan --province P` prints JSON with every task and a `done` flag, for dynamic task mapping. Every task is idempotent:
re-running a done task returns in seconds. Two tasks on the same unit at the same time are not guarded: one task per unit.

## 4. Resume rule

A stage is done when `<prefix>/_state/<stage>.json` exists and every file it lists is on S3 with the listed size.
The marker also stores the code hash, the models, run time and host. A site whose download is done but not its processing
fetches its raw files from S3 (fast, same region) instead of Earth Engine.

## 5. Requirements

Access: S3 read/write on the bucket; outbound HTTPS to Earth Engine (`earthengine-highvolume.googleapis.com`) with a
service-account key (store it as an S3 object or a secret and pass `ee_key_file`). No GCS, no Google Drive.
Python 3.11 + `requirements.txt` (pinned; keep `lightgbm==4.7.0`, the models were saved with it).

Measured on the JupyterHub pod (16 cores / 32 GB request), per district of ~50-120k acres:

| step | time | peak disk | notes |
|---|---|---|---|
| download | 5-10 min | 5-15 GB raw | network bound (~32 parallel Earth Engine requests) |
| process | 5-13 min | +stacks ~ raw size | CPU bound; uses every core |
| upload | 1-3 min | - | same region |

Memory: every step sizes its workers from the pod's memory request (`MEM_GUARANTEE`) or the free RAM, so a small pod only
runs slower. Plan >= 8 GB for a district; huge gap AOIs are split into tiles automatically. Disk: >= 40 GB free per site
running at the same time. A whole province (~20 units) on one 16-core pod: ~4-6 h; with parallel site tasks it is bounded
by Earth Engine (keep <= ~3 downloads at once per project quota).

## 6. Tests

`scripts/pipeline/test_pipeline.py` (see the file for what each test checks): plan / params, grid tiling, S3 marker
validation, resume after a kill, skip of done work, and bit-identical outputs versus the locked JupyterHub run.

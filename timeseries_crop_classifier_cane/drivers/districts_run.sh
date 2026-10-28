#!/bin/bash
# End-to-end for analysis districts: GEE -> GCS export, fetch, process (stacks, S1 15d with track alias, S2 dry, pheno),
# map (r2 + October rule + recall rule v5).  usage: drivers/districts_run.sh "name:DESro:ASCro" ...  e.g. "site119:62:172"
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P; export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 EE_MAX_ACTIVE=600
declare -A DES ASC
for a in "$@"; do IFS=: read n d s <<< "$a"; DES[$n]=$d; ASC[$n]=$s; done
python $P/scripts/site_paths.py place   # new district folders sites/<d>_2026 -> sites/2026/<Province>/<d>_2026
env_site() { S=$(python $P/scripts/site_paths.py dir ${1}_2026); export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_S1_TRACKS="${DES[$1]}:DESCENDING,${ASC[$1]}:ASCENDING"
  export S1_TRACKS=$(printf "RO%03d_DES,RO%03d_ASC" ${DES[$1]} ${ASC[$1]}); export S1_TRACK_ALIAS=$(printf "RO%03d_DES:RO164_DES,RO%03d_ASC:RO099_ASC" ${DES[$1]} ${ASC[$1]}); }
for n in "${!DES[@]}"; do env_site $n; echo "=== submit $n $(date +%T)"; python scripts/export_gcs.py submit 2>&1 | grep "done:\|Traceback"; done
for round in $(seq 1 40); do left=0
  for n in "${!DES[@]}"; do env_site $n; python scripts/export_gcs.py fetch 2>&1 | grep "fetch done\|FAILED" | sed "s/^/$n /"
    left=$((left+$(python -c "import pandas as pd; m=pd.read_csv('$S/data/manifest/export_tasks.csv'); print(int((m.state!='SUCCEEDED').sum()))"))); done
  echo "round $round: $left tasks not finished $(date +%T)"; [ $left -eq 0 ] && break; sleep 300; done
for n in "${!DES[@]}"; do env_site $n; echo "=== process $n $(date +%T)"
  { python -u scripts/build_stacks.py && python -u scripts/s1_composites.py 2026nov && python -u scripts/s2_dry_features.py 2026 $S && python -u scripts/pheno_features.py 2026 $S; } > $S/logs/process.log 2>&1 \
    && echo "PROCESSED $n" || { echo "FAILED $n (see $S/logs/process.log)"; continue; }
  rm -f $S/data/stacks/S1_*.tif $S/data/stacks/S2_cs.tif; rm -rf $S/data/raw/s1 $S/data/raw/s2 $S/data/raw/s2_cloud $S/data/raw/_gcs_tmp
  MODEL=data/model/s2on_r2/model.joblib OUT_TAG=s2on_r2 FEATURES_FILE=s1_15d_2026nov.tif YEAR=2026 python scripts/apply_site.py $S > $S/logs/apply_s2on_r2.log 2>&1 || { echo "apply FAILED $n"; continue; }
  python scripts/oct_bare_rule.py s2on_r2 s2on_r2_oct $S | tail -1; RULE=5 python scripts/recall_rule.py s2on_r2_oct s2on_r2_oct_rn5 $S | tail -1
done
echo ALL_DONE

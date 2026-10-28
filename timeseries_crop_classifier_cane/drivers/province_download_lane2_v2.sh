#!/bin/bash
# Second download lane for province_run_v2.sh (27 Oct 2026): takes districts from the END of the PLAN list (marker DL2_OWNER) and stops
# when it gets within 2 districts of lane D, so GEE fetches two districts at once while the process lane never waits.
# usage: PROV=province_le PLAN=sites/province6_plan.txt bash drivers/province_download_lane2_v2.sh
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P
export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 DL_WORKLOAD_TAG=standing-cane-${PROV,,} DL_S2_QA="" DL_CS_BANDS=cs_cdf
mapfile -t L < <(grep -v '^\s*$' ${PLAN:?})
for ((k=${#L[@]}-1; k>=3; k--)); do read n d a <<< "${L[$k]}"; S=$(python scripts/site_paths.py dir ${n}_2026)
  [ -f $S/DOWNLOADED ] && continue
  [ -f $S/logs/download_s1.log ] && [ ! -f $S/DL2_OWNER ] && { echo "D2: $n already started by lane D -> stop"; break; }
  read pn _ <<< "${L[$((k-3))]}"; prev=$(python scripts/site_paths.py dir ${pn}_2026); [ -f $prev/logs/download_s1.log ] && { echo "D2: lane D is close -> stop"; break; }
  touch $S/DL2_OWNER; D=$(printf "RO%03d_DES" $d); A=$(printf "RO%03d_ASC" $a)
  export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_S1_TRACKS="$d:DESCENDING,$a:ASCENDING"
  echo "D2: download $n $(date +%H:%M)"; t0=$(date +%s); ok=1
  for only in s1 s2; do for try in 1 2 3; do
      while [ $(df --output=avail -BG "$P" | tail -1 | tr -dc 0-9) -lt 35 ]; do sleep 60; done; DL_WORKERS=8 DL_BLOCK_THREADS=4 python -u scripts/download.py --only $only > $S/logs/download_$only.log 2>&1 && grep -q " 0 failed" $S/logs/download_$only.log && break; done
    grep -q " 0 failed" $S/logs/download_$only.log || ok=0; done
  [ $ok = 1 ] && { touch $S/DOWNLOADED; echo "D2: DOWNLOADED $n in $(( ($(date +%s)-t0)/60 )) min"; } || echo "D2: GAVE UP $n"
done; echo "D2: LANE DONE"

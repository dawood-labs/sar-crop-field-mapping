#!/bin/bash
# Second download lane for province13 (23 Oct): walks the district list from the LARGEST end while lane D walks from the smallest.
# Safety: takes district k only if lane D has not started k-2 yet (no download_s1.log), so the two lanes never meet in one folder.
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P
export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 DL_WORKLOAD_TAG=crop-province
declare -A ASC=([site36]=26 [site48]=26 [site40]=26 [site10]=26 [site26]=99 [site17]=99 [site22]=99 [site43]=26
  [site31]=99 [site52]=26 [site16]=99 [site28]=26 [site35]=99 [site02]=26 [site21]=99 [site07]=99)
ORDER=(site36 site48 site40 site10 site26 site17 site22 site43 site31 site52 site16 site28 site35 site02 site21 site07)
for ((k=${#ORDER[@]}-1; k>=2; k--)); do n=${ORDER[$k]}; S=$(python scripts/site_paths.py dir ${n}_2026)
  [ -f $S/DOWNLOADED ] && continue
  [ -f $S/logs/download_s1.log ] && [ ! -f $S/DL2_OWNER ] && { echo "D2: $n already started by lane D -> stop"; break; }
  prev=$(python scripts/site_paths.py dir ${ORDER[$((k-2))]}_2026); [ -f $prev/logs/download_s1.log ] && { echo "D2: lane D is close ($((k-2))) -> stop"; break; }
  a=${ASC[$n]}; export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_S1_TRACKS="164:DESCENDING,$a:ASCENDING"
  touch $S/DL2_OWNER; echo "D2: download $n (ASC RO$a) $(date +%H:%M)"; t0=$(date +%s); ok=1
  for only in s1 s2; do for try in 1 2 3; do
      DL_WORKERS=8 DL_BLOCK_THREADS=4 python -u scripts/download.py --only $only > $S/logs/download_$only.log 2>&1 && grep -q " 0 failed" $S/logs/download_$only.log && break
      echo "D2: $n $only try $try failed"; done; grep -q " 0 failed" $S/logs/download_$only.log || ok=0; done
  [ $ok = 1 ] && { touch $S/DOWNLOADED; echo "D2: DOWNLOADED $n in $(( ($(date +%s)-t0)/60 )) min, $(du -sh $S/data/raw | cut -f1)"; } || echo "D2: GAVE UP $n"
done; echo "D2: LANE DONE"

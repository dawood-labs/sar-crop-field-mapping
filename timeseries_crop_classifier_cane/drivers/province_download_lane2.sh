#!/bin/bash
# Second download lane for province13 (23 Oct): walks the district list from the LARGEST end while lane D walks from the smallest.
# Safety: takes district k only if lane D has not started k-2 yet (no download_s1.log), so the two lanes never meet in one folder.
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P
export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 DL_WORKLOAD_TAG=crop-province
declare -A ASC=([site116]=26 [site141]=26 [site130]=26 [site14]=26 [site87]=99 [site30]=99 [site80]=99 [site135]=26
  [site109]=99 [site169]=26 [site29]=99 [site103]=26 [site115]=99 [site02]=26 [site75]=99 [site10]=99)
ORDER=(site116 site141 site130 site14 site87 site30 site80 site135 site109 site169 site29 site103 site115 site02 site75 site10)
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

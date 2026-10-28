#!/bin/bash
# Field-delineation sugarcane polygons for the manager's review (27 Oct 2026): 4 versions per province, one run after another
#   v1 = cut every field below 75 % cane (first rule) ; 1ac / 1p5ac / 2ac = keep a field whole while its non-cane part <= X acres
# then a summary CSV + README and the upload to company Drive Standing_Cane_2026/<Province>/4_field_delineation_review/.
# Restart-safe: finished versions are skipped, RESUME=1 restarts from the saved cut checkpoint.
# usage: bash drivers/fd_review_province.sh province9 [province7 ...]
P="$(cd "$(dirname "$0")/.." && pwd)"; export PATH=$HOME/.local/bin:$PATH
for prov in "$@"; do p=$(echo $prov | tr A-Z a-z); D=$P/data/delineation/${p}_2026; O=$P/sites/2026/$prov/fd_review; mkdir -p $O
  F=$(ls $D/*_fields_2026.parquet); B=$(ls "$D"/*Boundary.gpkg); C=$P/sites/2026/$prov/${p}_class_2026_s20.tif
  for v in v1:-1 1ac:1 1p5ac:1.5 2ac:2; do tag=${v%%:*}; x=${v#*:}; o=$O/${p}_fd_sugarcane_2026_${tag}.gpkg
    [ -f ${o%.gpkg}_log.json ] && { echo "$prov $tag done"; continue; }
    echo "=== $prov $tag $(date +%H:%M)"
    (cd $P/scripts && RESUME=1 NONCANE_MAX_AC=$x NONCANE_INCL_CASSAVA=1 python -u fd_cane_fields.py "$F" "$B" "$C" $o > $P/logs/fd_${p}_${tag}.log 2>&1) \
      && echo "DONE $prov $tag $(date +%H:%M)" || echo "FAILED $prov $tag"
  done
  python $P/scripts/fd_review_summary.py $prov && rclone copy $O "company:Standing_Cane_2026/$prov/4_field_delineation_review/" --include "*.gpkg" --include "*.csv" --include "README.txt" --exclude "*precleaned*" --transfers 8 \
    && echo "UPLOADED $prov $(date +%H:%M)"
done; echo FD_REVIEW_DONE

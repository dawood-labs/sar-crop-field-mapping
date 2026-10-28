#!/bin/bash
# Province queue (27 Oct 2026): for each province: wait for its district run + gap run -> province outputs (mosaic + polygons, clipped)
# -> S3 + Drive -> field-delineation review (4 versions, low priority) ; the next province's district run starts as soon as the
# previous one's process lane is done. Restart-safe (each step skips finished work).
# usage: bash drivers/province_queue.sh province_le:62:99:a province5:164:99 province1:164:99      (gap orbits per province)
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P; A=$HOME/.config/standing_cane/aws.env; export PATH=$HOME/.local/bin:$PATH
for spec in "$@"; do IFS=: read prov des asc al <<< "$spec"; p=$(echo $prov | tr A-Z a-z)
  while pgrep -f "drivers/province_run_v2.sh" >/dev/null; do sleep 60; done     # one district run at a time (a running one finishes first)
  # district run (skips finished districts); returns when all districts are processed + uploaded
  echo "=== $prov districts $(date +%H:%M)"
  grep -q "P1: LANE DONE" logs/${p}_P1.log 2>/dev/null || PROV=$prov PLAN=sites/${p}_plan.txt LANE_CPUS=${Q_CPUS:-10} LANE_MEM=${Q_MEM:-14000000000} AWS_ENV=$A bash drivers/province_run_v2.sh >> logs/${p}_run.log 2>&1
  # gap (skips when done)
  echo "=== $prov gap $(date +%H:%M)"
  while pgrep -f "gap_run.sh \"${p}_gap_2026" >/dev/null || pgrep -fa gap_run.sh | grep -q ${p}_gap; do sleep 60; done
  [ -f sites/2026/$prov/${p}_gap_2026/UPLOADED ] || SC_CPUS=${Q_CPUS:-10} MEM_GUARANTEE=${Q_MEM:-14000000000} DL_CLOUD_BATCH=10 DL_AOI_SIMPLIFY=100 AWS_ENV=$A \
     bash drivers/gap_run.sh "${p}_gap_2026:$des:$asc${al:+:$al}" >> logs/${p}_gap_run.log 2>&1
  # province outputs
  echo "=== $prov province outputs $(date +%H:%M)"
  (cd scripts && python province_mosaic.py $prov data/model/s2on_r2_oct_rn5/cane_map_2026.tif ${p}_cane_map_2026.tif && python province_cane.py $prov) > logs/${p}_province.log 2>&1 \
    && ( source $A; B=s3://$S3_BUCKET/$S3_PREFIX/2026/$prov/province; for f in sites/2026/$prov/${p}_*.*; do aws s3 cp --only-show-errors $f $B/$(basename $f); done ) \
    && rclone copy sites/2026/$prov "company:Standing_Cane_2026/$prov/1_sugarcane_polygons/" --include "${p}_sugarcane_2026*" --max-depth 1 \
    && rclone copy sites/2026/$prov "company:Standing_Cane_2026/$prov/2_timeseries_cane_map/" --include "${p}_cane_map_2026.*" --max-depth 1 \
    && echo "PROVINCE DONE $prov $(date +%H:%M)" || echo "PROVINCE OUTPUT FAILED $prov"
  # field-delineation review in the background at low priority (one at a time)
  (while pgrep -f fd_review_province.sh >/dev/null; do sleep 120; done; nice -n 15 bash drivers/fd_review_province.sh $prov >> logs/fd_review_${p}.log 2>&1) &
done; wait; echo QUEUE_DONE

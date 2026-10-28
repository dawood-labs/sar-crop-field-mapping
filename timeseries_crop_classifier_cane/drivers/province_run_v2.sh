#!/bin/bash
# Generic province run (27 Oct 2026, from nblp_run.sh): PLAN file with lines '<district> <DES orbit> <ASC orbit>'; orbits other than the
# model's RO164 DES / RO099 ASC are aliased to those names. usage: PROV=province_le PLAN=sites/province6_plan.txt bash drivers/province_run_v2.sh
# Lane D (download): one district at a time, smallest first, 8 files x 4 blocks in parallel (~32 GEE requests) -> DOWNLOADED marker.
# Lanes P1/P2 (process): next downloaded district -> stacks -> S1 15d (2026nov) -> S2 dry -> pheno -> layer 1 (s2on_r2 + Oct rule +
# recall v5) -> layer 2 (colour v8 = layer2_final) -> S3 upload (raw + outputs, 32 streams, size-verified) -> delete raw + stacks.
# usage: setsid nohup bash sites/_drivers/province13_run.sh > logs/province_thani_run.log 2>&1 &     env AWS_ENV=<file with AWS creds>
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P
export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 DL_WORKLOAD_TAG=standing-cane-${PROV,,} DL_S2_QA="" DL_CS_BANDS=cs_cdf
S3=s3://$S3_BUCKET/$S3_PREFIX/2026/${PROV}/districts
# ASC track per district (coverage check 23 Oct): RO099 where it covers >= 98 %, else RO026 (aliased to the model's RO099 names)
declare -A DES ASC; ORDER=""
while read n d a; do [ -z "$n" ] && continue; DES[$n]=$d; ASC[$n]=$a; ORDER="$ORDER $n"; done < ${PLAN:?}
env_site() { S=$(python $P/scripts/site_paths.py dir ${1}_2026); d=${DES[$1]}; a=${ASC[$1]}
  D=$(printf "RO%03d_DES" $d); A=$(printf "RO%03d_ASC" $a)
  export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_S1_TRACKS="$d:DESCENDING,$a:ASCENDING" S1_TRACKS="$D,$A" S1_TRACK_ALIAS=""
  al=""; [ $D != RO164_DES ] && al="$D:RO164_DES"; [ $A != RO099_ASC ] && al="${al:+$al,}$A:RO099_ASC"; export S1_TRACK_ALIAS="$al"; }

download_lane() {
  for n in $ORDER; do env_site $n
    [ -f $S/DOWNLOADED ] && continue
    while [ $(df --output=avail -BG "$P" | tail -1 | tr -dc 0-9) -lt 40 ]; do echo "D: disk < 40 GB, waiting $(date +%H:%M)"; sleep 120; done
    echo "D: download $n (DES RO${DES[$n]} ASC RO${ASC[$n]}) $(date +%H:%M)"; t0=$(date +%s)
    ok=1
    for only in s1 s2; do                                # not s2x (red-edge extras are not used)
      for try in 1 2 3; do
        while [ $(df --output=avail -BG "$P" | tail -1 | tr -dc 0-9) -lt 35 ]; do sleep 60; done; DL_WORKERS=8 DL_BLOCK_THREADS=4 python -u scripts/download.py --only $only > $S/logs/download_$only.log 2>&1 && grep -q " 0 failed" $S/logs/download_$only.log && break
        echo "D: $n $only try $try: $(grep -c FAILED $S/logs/download_$only.log) failed files"; done
      grep -q " 0 failed" $S/logs/download_$only.log || ok=0; done
    [ $ok = 1 ] && { touch $S/DOWNLOADED; echo "D: DOWNLOADED $n in $(( ($(date +%s)-t0)/60 )) min, $(du -sh $S/data/raw | cut -f1)"; } || echo "D: GAVE UP $n"
  done; echo "D: LANE DONE"
}

process_lane() {   # $1 = lane name
  export SC_CPUS=${LANE_CPUS:-10} MEM_GUARANTEE=${LANE_MEM:-12884901888}   # one lane, 12 GB (FD job runs beside it)
  while true; do
    left=0; took=""
    for n in $ORDER; do env_site $n
      [ -f $S/data/model/layer2_final/class3_2026.tif ] && [ -f $S/UPLOADED ] && continue; left=1
      [ -f $S/DOWNLOADED ] || continue
      mkdir $S/PROCESSING 2>/dev/null || continue; took=$n; break; done
    [ $left -eq 0 ] && { echo "$1: LANE DONE"; return; }
    [ -z "$took" ] && { sleep 60; continue; }
    n=$took; env_site $n; t0=$(date +%s); echo "$1: process $n $(date +%H:%M)"
    if [ ! -f $S/data/model/layer2_final/class3_2026.tif ]; then
      { python -u scripts/build_stacks.py && python -u scripts/s1_composites.py 2026nov && python -u scripts/s2_dry_features.py 2026 $S \
        && python -u scripts/pheno_features.py 2026 $S \
        && MODEL=data/model/s2on_r2/model.joblib OUT_TAG=s2on_r2 FEATURES_FILE=s1_15d_2026nov.tif YEAR=2026 python scripts/apply_site.py $S \
        && python scripts/oct_bare_rule.py s2on_r2 s2on_r2_oct $S && RULE=5 python scripts/recall_rule.py s2on_r2_oct s2on_r2_oct_rn5 $S \
        && COLOUR_TAG=v8 OUT_TAG=layer2_final SMOOTH=9 LAYER1=s2on_r2_oct_rn5 python scripts/colour_multi.py apply ${n}_2026; } > $S/logs/process.log 2>&1 \
        || { echo "$1: FAILED $n (see $S/logs/process.log)"; rmdir $S/PROCESSING; touch $S/FAILED; continue; }
      echo "$1: PROCESSED $n in $(( ($(date +%s)-t0)/60 )) min: $(grep -o "cane_ac_total': [0-9]*" $S/data/model/s2on_r2_oct_rn5/rule.json 2>/dev/null)"
    fi
    ( source ${AWS_ENV:?}; for s in raw stacks aoi grid manifest model features; do aws s3 sync --only-show-errors $S/data/$s $S3/${n}_2026/data/$s; done
      aws s3 ls --recursive $S3/${n}_2026/data/ > $S/logs/s3_list.txt ) \
      && python - "$S" "$n" <<'EOF' && { rm -rf $S/data/raw $S/data/stacks; touch $S/UPLOADED; echo "$1: UPLOADED + cleaned $n $(date +%H:%M) | disk free $(df -h "$P" | awk 'NR==2{print $4}')"; } || echo "$1: S3 UPLOAD/VERIFY FAILED $n (raw kept)"
import os, sys
S, n = sys.argv[1], sys.argv[2]
r = {}
for l in open(f"{S}/logs/s3_list.txt"):
    p = l.split(None, 3)
    if len(p) == 4: r[p[3].strip().split(f"/districts/{n}_2026/", 1)[1]] = int(p[2])
loc = [os.path.join(dp, f) for s in ("raw", "stacks", "aoi", "grid", "manifest", "model", "features") for dp, _, fs in os.walk(f"{S}/data/{s}") for f in fs]
bad = [f for f in loc if r.get(os.path.relpath(f, S)) != os.path.getsize(f)]
print(f"verify {n}: {len(loc)} files, {len(bad)} bad"); sys.exit(1 if bad else 0)
EOF
    rmdir $S/PROCESSING
  done
}

python scripts/site_paths.py place
download_lane > logs/${PROV,,}_D.log 2>&1 &
process_lane P1 > logs/${PROV,,}_P1.log 2>&1 &
wait; echo ALL_DONE $(date +%H:%M)

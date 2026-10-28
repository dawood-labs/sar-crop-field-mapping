#!/bin/bash
# province13 province (user, 23 Oct 2026): 16 districts (Site62 + site_ns already done), route B = GEE direct read.
# Lane D (download): one district at a time, smallest first, 8 files x 4 blocks in parallel (~32 GEE requests) -> DOWNLOADED marker.
# Lanes P1/P2 (process): next downloaded district -> stacks -> S1 15d (2026nov) -> S2 dry -> pheno -> layer 1 (s2on_r2 + Oct rule +
# recall v5) -> layer 2 (colour v8 = layer2_final) -> S3 upload (raw + outputs, 32 streams, size-verified) -> delete raw + stacks.
# usage: setsid nohup bash sites/_drivers/province13_run.sh > logs/province_thani_run.log 2>&1 &     env AWS_ENV=<file with AWS creds>
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P
export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 DL_WORKLOAD_TAG=crop-province
S3=s3://$S3_BUCKET/$S3_PREFIX/2026/province13/districts
# ASC track per district (coverage check 23 Oct): RO099 where it covers >= 98 %, else RO026 (aliased to the model's RO099 names)
declare -A ASC=([site96]=26 [site121]=26 [site110]=26 [site14]=26 [site67]=99 [site30]=99 [site60]=99 [site115]=26
  [site89]=99 [site149]=26 [site29]=99 [site83]=26 [site95]=99 [site02]=26 [site55]=99 [site10]=99)
ORDER="site96 site121 site110 site14 site67 site30 site60 site115 site89 site149 site29 site83 site95 site02 site55 site10"
env_site() { S=$(python $P/scripts/site_paths.py dir ${1}_2026); a=${ASC[$1]}
  export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_S1_TRACKS="164:DESCENDING,$a:ASCENDING" S1_TRACKS="RO164_DES,RO0${a}_ASC"
  [ $a = 99 ] && export S1_TRACKS="RO164_DES,RO099_ASC" S1_TRACK_ALIAS="" || export S1_TRACK_ALIAS="RO026_ASC:RO099_ASC"; }

download_lane() {
  for n in $ORDER; do env_site $n
    [ -f $S/DOWNLOADED ] && continue
    while [ $(df --output=avail -BG "$P" | tail -1 | tr -dc 0-9) -lt 40 ]; do echo "D: disk < 40 GB, waiting $(date +%H:%M)"; sleep 120; done
    echo "D: download $n (ASC RO${ASC[$n]}) $(date +%H:%M)"; t0=$(date +%s)
    ok=1
    for only in s1 s2; do                                # not s2x (red-edge extras are not used)
      for try in 1 2 3; do
        DL_WORKERS=8 DL_BLOCK_THREADS=4 python -u scripts/download.py --only $only > $S/logs/download_$only.log 2>&1 && grep -q " 0 failed" $S/logs/download_$only.log && break
        echo "D: $n $only try $try: $(grep -c FAILED $S/logs/download_$only.log) failed files"; done
      grep -q " 0 failed" $S/logs/download_$only.log || ok=0; done
    [ $ok = 1 ] && { touch $S/DOWNLOADED; echo "D: DOWNLOADED $n in $(( ($(date +%s)-t0)/60 )) min, $(du -sh $S/data/raw | cut -f1)"; } || echo "D: GAVE UP $n"
  done; echo "D: LANE DONE"
}

process_lane() {   # $1 = lane name
  export SC_CPUS=$(( $(nproc) / 2 )) MEM_GUARANTEE=$(( ${MEM_GUARANTEE:-38654705664} / 2 ))
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
download_lane > logs/province_D.log 2>&1 &
process_lane P1 > logs/province_P1.log 2>&1 &
sleep 30; process_lane P2 > logs/province_P2.log 2>&1 &
wait; echo ALL_DONE $(date +%H:%M)

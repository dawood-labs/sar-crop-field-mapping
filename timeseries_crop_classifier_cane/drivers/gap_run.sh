#!/bin/bash
# Province GAP areas (user, 23 Oct 2026): parts of a province that no GAUL district AOI covers, run as extra sites with the same
# recipe as the districts (route B download -> stacks -> features -> layer 1 -> layer 2 -> S3 upload of raw + stacks + outputs).
# Gap AOIs are thin / scattered, so download.py uses 256 px blocks and skips blocks without AOI pixels.
# usage: AWS_ENV=<creds file> setsid nohup bash drivers/gap_run.sh "site:DES:ASC[:alias]" ... > logs/gap_run.log 2>&1 &
#   e.g. "site51_2026:164:99" "site50_2026:164:26:a" "site19_2026:62:172:a"   (a = alias tracks to the model's)
P="$(cd "$(dirname "$0")/.." && pwd)"; cd $P
export GDAL_NUM_THREADS=ALL_CPUS DL_START=2025-11-01 DL_END=2026-10-15 DL_WORKLOAD_TAG=standing-cane-gaps DL_BLOCK=256 DL_S2_QA="" DL_CS_BANDS=cs_cdf   # only what the model + review images need
for spec in "$@"; do IFS=: read n des asc al <<< "$spec"
  S=$(python scripts/site_paths.py dir $n); prov=$(basename $(dirname $S))
  export SC_ROOT=$S SC_AOI=$S/data/aoi/aoi.shp DL_S1_TRACKS="$des:DESCENDING,$asc:ASCENDING"
  D=$(printf "RO%03d_DES" $des); A=$(printf "RO%03d_ASC" $asc); export S1_TRACKS="$D,$A" S1_TRACK_ALIAS=""
  [ -n "$al" ] && export S1_TRACK_ALIAS="$D:RO164_DES,$A:RO099_ASC"; [ "$D" = RO164_DES ] && [ -n "$al" ] && export S1_TRACK_ALIAS="$A:RO099_ASC"
  echo "=== $n ($prov) DES $D ASC $A alias '$S1_TRACK_ALIAS' $(date +%H:%M)"; t0=$(date +%s)
  if [ ! -f $S/DOWNLOADED ]; then ok=1
    for only in s1 s2; do for try in 1 2 3; do
        DL_WORKERS=8 DL_BLOCK_THREADS=4 python -u scripts/download.py --only $only > $S/logs/download_$only.log 2>&1 && grep -q " 0 failed" $S/logs/download_$only.log && break
        echo "$n $only try $try failed"; done; grep -q " 0 failed" $S/logs/download_$only.log || ok=0; done
    [ $ok = 1 ] && touch $S/DOWNLOADED || { echo "GAVE UP $n"; continue; }
    echo "DOWNLOADED $n in $(( ($(date +%s)-t0)/60 )) min, $(du -sh $S/data/raw | cut -f1)"; fi
  if [ ! -f $S/data/model/layer2_final/class3_2026.tif ]; then
    { python -u scripts/build_stacks.py && python -u scripts/s1_composites.py 2026nov && python -u scripts/s2_dry_features.py 2026 $S \
      && python -u scripts/pheno_features.py 2026 $S \
      && MODEL=data/model/s2on_r2/model.joblib OUT_TAG=s2on_r2 FEATURES_FILE=s1_15d_2026nov.tif YEAR=2026 python scripts/apply_site.py $S \
      && python scripts/oct_bare_rule.py s2on_r2 s2on_r2_oct $S && RULE=5 python scripts/recall_rule.py s2on_r2_oct s2on_r2_oct_rn5 $S \
      && COLOUR_TAG=v8 OUT_TAG=layer2_final SMOOTH=9 LAYER1=s2on_r2_oct_rn5 python scripts/colour_multi.py apply $n; } > $S/logs/process.log 2>&1 \
      || { echo "FAILED $n (see $S/logs/process.log)"; continue; }
    echo "PROCESSED $n in $(( ($(date +%s)-t0)/60 )) min"; fi
  B=s3://$S3_BUCKET/$S3_PREFIX/2026/$prov/districts/$n/data
  ( source ${AWS_ENV:?}; for s in raw stacks aoi grid manifest model features; do aws s3 sync --only-show-errors $S/data/$s $B/$s; done
    aws s3 ls --recursive $B/ > $S/logs/s3_list.txt ) && python - "$S" <<'EOF' && { rm -rf $S/data/raw $S/data/stacks; touch $S/UPLOADED; echo "UPLOADED + cleaned $n $(date +%H:%M)"; } || echo "S3 UPLOAD/VERIFY FAILED $n (raw kept)"
import os, sys
S = sys.argv[1]; r = {}
for l in open(f"{S}/logs/s3_list.txt"):
    p = l.split(None, 3)
    if len(p) == 4: r[p[3].strip().split("/data/", 1)[1]] = int(p[2])
loc = [os.path.join(dp, f) for s in ("raw", "stacks", "aoi", "grid", "manifest", "model", "features") for dp, _, fs in os.walk(f"{S}/data/{s}") for f in fs]
bad = [f for f in loc if r.get(os.path.relpath(f, f"{S}/data")) != os.path.getsize(f)]
print(f"verify {os.path.basename(S)}: {len(loc)} files, {len(bad)} bad"); sys.exit(1 if bad else 0)
EOF
done; echo GAPS_DONE $(date +%H:%M)

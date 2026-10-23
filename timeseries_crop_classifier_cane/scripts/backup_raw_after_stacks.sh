#!/bin/bash
# Wait until every stack is built (no build_stacks process, stacks.csv written, no .tmp), then back up data/raw to GCS,
# verify and delete the local raw files. Manifests (data/manifest/*.csv) stay local.
cd "$(dirname "$0")/.."
while ps -eo args | grep -q "[p]ython scripts/build_stacks.py"; do sleep 20; done
if [ ! -f data/stacks/stacks.csv ] || ls data/stacks/*.tmp >/dev/null 2>&1; then echo "stacks not complete, raw kept"; exit 1; fi
n=$(ls data/stacks/*.tif | wc -l); echo "stacks: $n"
python scripts/upload.py data/raw --delete

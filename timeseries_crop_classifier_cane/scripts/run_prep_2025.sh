#!/bin/bash
# 2025 preparation chain: S2 stacks and S1 15-day composites in parallel, then S2 NDVI / clear mask, then EDA.
cd "$(dirname "$0")/.."
python scripts/build_stacks.py > logs/build_stacks.log 2>&1 &
A=$!
python scripts/s1_composites.py 2025 > logs/s1_composites_2025.log 2>&1 &
B=$!
wait $A; wait $B
python scripts/precompute.py s2 > logs/precompute_s2.log 2>&1
python scripts/eda.py > logs/eda.log 2>&1
echo "chain done" >> logs/eda.log

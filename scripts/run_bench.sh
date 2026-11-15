#!/usr/bin/env bash
# Runs the benchmark binary with the project's standard measurement settings and saves
# all repetitions as JSON in results/ (the console shows only mean/median/stddev/cv).
#
# Usage:   bash scripts/run_bench.sh <name> [filter regex] [repetitions]
# Example: bash scripts/run_bench.sh stage1_naive 'BM_Matmul/naive' 5
set -euo pipefail

cd "$(dirname "$0")/.."

name="${1:?usage: run_bench.sh <name> [filter regex] [repetitions]}"
filter="${2:-.}"
reps="${3:-5}"

# Record which code produced the numbers; "-dirty" means uncommitted changes.
commit="$(git log -1 --format=%h 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    commit="${commit}-dirty"
fi

mkdir -p results
./build/matmul_bench \
    --benchmark_filter="$filter" \
    --benchmark_repetitions="$reps" \
    --benchmark_display_aggregates_only=true \
    --benchmark_context=commit="$commit" \
    --benchmark_out="results/${name}.json" \
    --benchmark_out_format=json

echo "Saved results/${name}.json"

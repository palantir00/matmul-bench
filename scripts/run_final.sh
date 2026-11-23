#!/usr/bin/env bash
# Final measurement series for the README: every C++ benchmark plus the PyTorch reference,
# with the clock probed before and after. Takes ~10 minutes.
#
# Usage: bash scripts/run_final.sh [reps=5]
#   Run from WSL/Linux with the Python venv active (scripts/requirements.txt), after a Release
#   build. Before running: AC power, best-performance power mode, background programs closed.
set -euo pipefail
cd "$(dirname "$0")/.."

reps="${1:-5}"
python="${PYTHON:-python3}"
mkdir -p results

{
    echo "clock before:"
    ./build/clock_probe | tail -3
} | tee results/final_clock.txt

bash scripts/run_bench.sh final_cpp '.' "$reps"
"$python" scripts/bench_torch.py --reps "$reps" --out results/final_torch.json

{
    echo "clock after:"
    ./build/clock_probe | tail -3
} | tee -a results/final_clock.txt

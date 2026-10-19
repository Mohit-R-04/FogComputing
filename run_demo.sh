#!/usr/bin/env bash
#
# Terminal demonstration of the whole project.
#
#   bash run_demo.sh             quick (under a minute): reuses generated data and saved
#                                experiment results, runs the live parts (build, training,
#                                the BBN controlling iFogSim, all policies on one run)
#   bash run_demo.sh --full      regenerates everything first (about 45 minutes)
#   bash run_demo.sh --no-pause  the live run without pacing
#
# Requirements: Java 17+ (javac, java) and Python 3.10+. The Python environment is
# created on first use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x model/.venv/bin/python ]; then
    echo "creating the Python environment (first run only) ..."
    ( cd model && bash setup_venv.sh )
fi
if [ ! -f data/bbn_trained.json ] || [ ! -f model/comparison/results/comparison_summary.json ] \
        || [ ! -f model/results/ft/summary.json ]; then
    echo "no saved results yet: running the full pipeline once (bash run_all.sh) ..."
    bash run_all.sh
fi
exec model/.venv/bin/python model/demo.py "$@"

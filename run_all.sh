#!/usr/bin/env bash
#
# One command to run the whole project: iFogSim simulation -> dataset -> BBN.
# Run from the solution/ root (the script roots itself there).
#
#   bash run_all.sh
#
# See README.md for what each step does.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "1/6  compile the telemetry extension (ifogsim/)"
( cd ifogsim && bash build.sh )

step "2/6  generate telemetry with iFogSim (12 scenarios)"
( cd ifogsim && bash run_sweep.sh )

step "3/6  build the dataset"
if [ ! -x model/.venv/bin/python ]; then
    echo "creating the Python environment first"
    ( cd model && bash setup_venv.sh )
fi
( cd model && .venv/bin/python build_dataset.py --cases 200 --holdout 5 --seed 999 )

step "4/6  train the BBN"
( cd model && .venv/bin/python train_bbn.py )

step "5/6  predict (elicited, then trained)"
( cd model && .venv/bin/python predict.py )
( cd model && .venv/bin/python predict.py --learned ../data/cpts_trained.json \
      --out results/predictions_trained.csv >/dev/null )

step "6/6  verify"
( cd model && .venv/bin/python verify.py )

printf '\n\033[1mdone.\033[0m\n'
echo "  data/raw/scenario_*.csv            telemetry produced by iFogSim"
echo "  data/telemetry_cases.csv           the labelled evaluation cases"
echo "  data/cpts_trained.json             the trained network"
echo "  model/results/predictions.csv       posteriors, elicited"
echo "  model/results/predictions_trained.csv  posteriors, trained"
echo "  model/results/verification_report.md   the full report"

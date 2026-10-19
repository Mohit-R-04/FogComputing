#!/usr/bin/env bash
#
# The whole project, reproducibly, in one command:
#
#   bash run_all.sh            # everything (about 45 minutes)
#   bash run_all.sh --skip-selection   # reuse data/models/bbn_config.json
#
#   1  compile the iFogSim extension                      (ifogsim/build.sh)
#   2  generate labelled telemetry with iFogSim           (31 runs, parallel JVMs)
#   3  build the training and held-out pools              (build_dataset.py)
#   4  choose the BBN configuration on training folds     (select_bbn.py)
#   5  train the BBN                                      (train_bbn.py)
#   6  verify it on the held-out runs                     (verify.py, predict.py)
#   7  compare it with eight alternatives                 (comparison/run_comparison.py)
#   8  closed-loop fault tolerance inside iFogSim         (run_ft_experiments.py)
#   9  tests: inference, pipeline, Java/Python parity, pgmpy cross-check
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SKIP_SELECTION=0
for arg in "$@"; do
    [ "$arg" = "--skip-selection" ] && SKIP_SELECTION=1
done

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

if [ ! -x model/.venv/bin/python ]; then
    step "0/9  create the Python environment"
    ( cd model && bash setup_venv.sh )
fi
PY="$ROOT/model/.venv/bin/python"

step "1/9  compile the iFogSim extension"
( cd ifogsim && bash build.sh )

step "2/9  generate telemetry with iFogSim (12 regimes x seeds, 31 runs)"
( cd model && "$PY" generate_telemetry.py )

step "3/9  build the dataset"
( cd model && "$PY" build_dataset.py )

if [ "$SKIP_SELECTION" -eq 0 ] || [ ! -f data/models/bbn_config.json ]; then
    step "4/9  select the BBN configuration (training scenarios only)"
    ( cd model && "$PY" -u select_bbn.py )
else
    step "4/9  reuse the selected BBN configuration"
    cat data/models/bbn_config.json
fi

step "5/9  train the BBN"
( cd model && "$PY" train_bbn.py --quiet )

step "6/9  verify the BBN on held-out runs"
( cd model && "$PY" predict.py && "$PY" predict.py --elicited && "$PY" verify.py )

step "7/9  compare with alternative models"
( cd model && "$PY" -u comparison/run_comparison.py )

step "8/9  closed-loop fault tolerance in iFogSim"
( cd model && "$PY" -u run_ft_experiments.py )

step "9/9  tests"
( cd model && "$PY" -m pytest -q test_inference.py test_pipeline.py test_java_parity.py && "$PY" crosscheck_pgmpy.py | tail -2 )

printf '\n\033[1mdone.\033[0m\n'
echo "  data/raw/scenario_SS_seed_K.csv           telemetry produced by iFogSim"
echo "  data/bbn_trained.json                     the trained network (Python and Java)"
echo "  model/results/verification_report.md      BBN verification on held-out runs"
echo "  model/comparison/results/model_comparison.md   BBN vs eight alternatives"
echo "  model/results/ft/policy_comparison.md     closed-loop fault tolerance results"
echo "  bash run_demo.sh                          the terminal demonstration"

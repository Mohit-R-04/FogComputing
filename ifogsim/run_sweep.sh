#!/usr/bin/env bash
#
# Java-only telemetry sweep: every (or selected) scenario for one seed, one JVM each,
# writing labelled snapshots to ../data/raw/scenario_SS_seed_K.csv.
#
#   ./run_sweep.sh                    # all 12 scenarios, seed 101
#   ./run_sweep.sh 0 3 7              # only these scenarios
#   SWEEP_SEED=201 ./run_sweep.sh     # another seed
#
# The full experimental design (training and held-out seeds, run in parallel) is
# model/generate_telemetry.py; this script is for running the simulator by hand.
#
# One JVM per scenario because CloudSim keeps its clock, entity list and event queue in
# statics. Run from inside ifogsim/: the simulator resolves ./dataset/... relative to
# the working directory (the script roots itself there).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CLASSES="build/classes"
CP="$CLASSES:jars/*:jars/commons-math3-3.5/*"
RAW="../data/raw"
SEED="${SWEEP_SEED:-101}"
SIM_TIME="${SIM_TIME:-10000}"

if [ ! -f "$CLASSES/org/fog/ft/ScenarioSweep.class" ]; then
    echo "Not compiled yet. Run ./build.sh first." >&2
    exit 1
fi

COUNT=$(java -cp "$CP" org.fog.ft.ScenarioSweep --list | head -1 | grep -o '[0-9]*$')
if [ $# -gt 0 ]; then
    SCENARIOS=("$@")
else
    SCENARIOS=()
    for ((i = 0; i < COUNT; i++)); do SCENARIOS+=("$i"); done
fi

mkdir -p "$RAW"
echo "running ${#SCENARIOS[@]} of $COUNT scenarios, seed $SEED, sim time $SIM_TIME"
for i in "${SCENARIOS[@]}"; do
    OUT=$(printf "%s/scenario_%02d_seed_%s.csv" "$RAW" "$i" "$SEED")
    java -cp "$CP" org.fog.ft.ScenarioSweep "$i" "$SEED" "$OUT" "$SIM_TIME" \
        | grep -E '^=== scenario|^failures|^telemetry' || true
done

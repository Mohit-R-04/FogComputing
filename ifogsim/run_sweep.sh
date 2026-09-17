#!/usr/bin/env bash
#
# Run the telemetry sweep: every scenario of the iFogSim simulation, one JVM each,
# writing the labelled snapshots to ../data/raw/scenario_NN.csv.
#
#   ./run_sweep.sh                 # all 12 scenarios
#   ./run_sweep.sh 0 3 7           # only these three
#   SWEEP_SEED=12345 ./run_sweep.sh
#
# One JVM per scenario because CloudSim keeps its clock, entity list and event
# queue in statics, and iFogSim's Controller calls System.exit(0) on STOP_SIMULATION.
#
# Run from inside ifogsim/: the simulator's DataParser resolves ./dataset/...
# relative to the working directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CLASSES="build/classes"
CP="$CLASSES:jars/*:jars/commons-math3-3.5/*"
RAW="../data/raw"
SEED="${SWEEP_SEED:-20260912}"

if [ ! -d "$CLASSES/org/fog/ft" ]; then
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
echo "running ${#SCENARIOS[@]} of $COUNT scenarios, seed $SEED"
echo "output: $RAW/"
echo

for i in "${SCENARIOS[@]}"; do
    OUT=$(printf "%s/scenario_%02d.csv" "$RAW" "$i")
    printf -- '-- scenario %s --\n' "$i"
    java -cp "$CP" org.fog.ft.ScenarioSweep "$i" "$SEED" "$OUT" \
        | grep -E 'scenario [0-9]+ \(|snapshots written|nodes that failed|positive snapshots|rows reported' \
        || true
    echo
done

echo "done. files in $RAW:"
ls -la "$RAW"

#!/usr/bin/env bash
#
# Compile the telemetry extension (org.fog.ft) against this iFogSim checkout.
# Run from inside the ifogsim/ folder (the script roots itself there).
#
# The extension is a new package, src/org/fog/ft/, added on top of the upstream
# source tree. Nothing upstream is modified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p build/classes
javac -nowarn \
      -cp "jars/*:jars/commons-math3-3.5/*" \
      -sourcepath src \
      -d build/classes \
      src/org/fog/ft/*.java

echo "compiled $(find build/classes -name '*.class' | wc -l | tr -d ' ') classes into build/classes"

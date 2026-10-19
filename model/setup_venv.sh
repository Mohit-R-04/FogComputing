#!/usr/bin/env bash
# Creates the Python environment for the BBN, the comparison suite and the demo.
#
# A virtual environment is not optional here. Homebrew's Python is marked
# externally managed (PEP 668), so installing into it is refused.
#
# Python 3.12 is preferred when it is installed because every dependency
# (including the optional pgmpy cross-check) publishes wheels for it; any
# Python >= 3.10 works for the core pipeline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

pick_python() {
    for candidate in python3.12 python3.13 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return
        fi
    done
    echo "no python3 interpreter found" >&2
    exit 1
}

if [ ! -d "$VENV" ]; then
    PY="$(pick_python)"
    echo "creating virtualenv at $VENV with $($PY --version)"
    "$PY" -m venv "$VENV"
fi

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$HERE/requirements.txt" || {
    echo "full install failed; retrying without the optional pgmpy cross-check" >&2
    grep -v '^pgmpy' "$HERE/requirements.txt" > "$VENV/requirements-core.txt"
    "$VENV/bin/pip" install --quiet -r "$VENV/requirements-core.txt"
}

echo "environment ready:"
"$VENV/bin/python" -c "import numpy, pandas, sklearn, matplotlib, rich; print('  core packages ok')"
"$VENV/bin/python" -c "import pgmpy; print('  pgmpy', pgmpy.__version__, '(cross-check available)')" 2>/dev/null \
    || echo "  pgmpy not installed (cross-check will skip)"

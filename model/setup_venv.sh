#!/usr/bin/env bash
# Creates the Python environment for the BBN.
#
# A virtual environment is not optional here. Homebrew's Python is marked
# externally managed (PEP 668), so installing into it is refused.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

if [ ! -d "$VENV" ]; then
    echo "creating virtualenv at $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$HERE/requirements.txt"

echo "environment ready:"
"$VENV/bin/python" -c "import numpy, pandas, sklearn, matplotlib; print('  core packages ok')"
"$VENV/bin/python" -c "import pgmpy; print('  pgmpy', pgmpy.__version__, '(cross-check available)')" 2>/dev/null \
    || echo "  pgmpy not installed (cross-check will skip)"

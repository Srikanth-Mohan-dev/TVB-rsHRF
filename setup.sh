#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv-develop-rshrf"

echo "============================================================"
echo "Setting up environment in $SCRIPT_DIR"
echo "(venv-develop-rshrf -- separate from the main venv)"
echo "============================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found on PATH. Install Python 3.9+ (e.g. via your"
    echo "system package manager, or https://python.org)."
    exit 1
fi

echo "[1/4] Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/4] Upgrading pip"
python -m pip install --upgrade pip

echo "[3/4] Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

echo "[4/4] Checking for AWS CLI"
if command -v aws >/dev/null 2>&1; then
    echo "  aws CLI already available: $(aws --version 2>&1)"
else
    echo "  aws CLI not found on PATH after pip install awscli."
    echo "  The pip package installs an \"aws\" command inside the venv;"
    echo "  if it's still not found, install the official AWS CLI v2 manually:"
    echo "    https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
fi

mkdir -p datasets results

echo
echo "============================================================"
echo "Done. To use this environment:"
echo "  source venv-develop-rshrf/bin/activate"
echo "  python pipeline.py --dataset ds001226 --subject CON01"
echo "============================================================"
echo
echo "NOTE: your original 'venv' folder (if it exists) is untouched --"
echo "this script only creates/uses 'venv-develop-rshrf'."
#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
TVB_DIR="$SCRIPT_DIR/tvb-root"

echo "============================================================"
echo "Setting up environment in $SCRIPT_DIR"
echo "============================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found on PATH. Install Python 3.9+ (e.g. via your"
    echo "system package manager, or https://python.org)."
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "git not found on PATH. Install git (e.g. via your system"
    echo "package manager, or https://git-scm.com/downloads) -- it's"
    echo "needed to clone tvb-root."
    exit 1
fi

echo "[1/5] Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Upgrading pip"
python -m pip install --upgrade pip

echo "[3/5] Installing Python dependencies from requirements.txt"
pip install -r requirements.txt

echo "[4/5] Cloning and installing TVB (hybrid-numba branch)"
if [ ! -d "$TVB_DIR" ]; then
    git clone --branch hybrid-numba https://github.com/the-virtual-brain/tvb-root.git "$TVB_DIR"
else
    echo "  $TVB_DIR already exists — skipping clone."
fi
pip install -e "$TVB_DIR/tvb_library"

echo "[5/5] Checking for AWS CLI"
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
echo "  source venv/bin/activate"
echo "  python pipeline.py --dataset ds001226 --subject CON01"
echo "============================================================"
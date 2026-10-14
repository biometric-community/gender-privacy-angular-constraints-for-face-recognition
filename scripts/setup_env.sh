#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REPO_ROOT="$ROOT"
while [[ "$REPO_ROOT" != "/" ]]; do
  if [[ -f "$REPO_ROOT/docs/PAPERS.md" ]] || [[ -d "$REPO_ROOT/.venv" ]]; then break; fi
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
SHARED_VENV="${TBIOT_SHARED_VENV:-$REPO_ROOT/.venv}"
# shellcheck disable=SC1091
source "$SHARED_VENV/bin/activate"
pip install -q -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

#!/bin/zsh
set -e
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
if [[ -x ../.venv/bin/python ]]; then
  PYTHON_BIN=../.venv/bin/python
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  echo "Python environment missing. Follow the setup in README.md."
  exit 1
fi
"$PYTHON_BIN" -m src.download_analyst_data --start 2000-01-01 --end 2025-12-31 "$@"
echo "Finished."

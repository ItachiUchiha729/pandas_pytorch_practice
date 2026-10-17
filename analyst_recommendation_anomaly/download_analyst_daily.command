#!/bin/zsh
set -e
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
../.venv/bin/python -m src.download_analyst_market --start 2000-01-01 --end 2024-12-31 "$@"

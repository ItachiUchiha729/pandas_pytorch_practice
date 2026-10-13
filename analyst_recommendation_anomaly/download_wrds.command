#!/bin/zsh

set -e

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

echo "Downloading IBES and CRSP data from WRDS into data/raw/"
echo "Existing Parquet files will be reused automatically."
echo

../.venv/bin/python -m src.download_data \
  --start 2000-01-01 \
  --end 2025-12-31

echo
echo "WRDS download finished. You may close this window."
read "?Press Enter to close..."

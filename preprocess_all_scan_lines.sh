#!/bin/bash
set -e

# Usage: ./preprocess_all_scan_lines.sh [key=value ...] [--force]

# Create output directories if they don't exist
mkdir -p processed_data cnn_preprocessed

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Error: venv directory not found. Please create it and install dependencies."
    exit 1
fi

PYTHON_BIN="venv/bin/python"

# Get list of folders to process
echo "Fetching list of folders..."
scan_folders=$($PYTHON_BIN -c "from common_utils import get_list_of_folders; print('\n'.join(get_list_of_folders()))" "$@")

if [ -z "$scan_folders" ]; then
    echo "No folders found to process."
    exit 0
fi

count=$(echo "$scan_folders" | wc -l)
echo "Found $count folders to process."

start_ts=$(date +%s)
# Run processing in parallel
# Using --no-notice to suppress the citation notice
parallel --no-notice --bar --eta -j 20 $PYTHON_BIN process_single.py {} "$@" :::: <(echo "$scan_folders")

echo "Elapsed: $(( $(date +%s) - start_ts ))s"
echo "All processing complete"

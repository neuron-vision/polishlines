#!/bin/bash

rm -rf processed_data
mkdir -p processed_data
scan_folders=$(venv/bin/python -c "from common_utils import get_list_of_folders; print('\n'.join(get_list_of_folders()))" "$@")
start_ts=$(date +%s)
parallel --bar --eta -j 20 venv/bin/python process_single_merged.py {} "$@" :::: <(echo "$scan_folders")
echo "Elapsed: $(( $(date +%s) - start_ts ))s"

outputs=$(venv/bin/python post_analysis.py "$@")

echo "All processing complete"

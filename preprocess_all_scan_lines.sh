#!/bin/bash

rm -rf processed_data cnn_preprocessed
mkdir -p processed_data cnn_preprocessed
scan_folders=$(venv/bin/python -c "from common_utils import get_list_of_folders; print('\n'.join(get_list_of_folders()))" "$@")
start_ts=$(date +%s)
parallel --bar --eta -j 20 venv/bin/python process_single.py {} "$@" :::: <(echo "$scan_folders")
echo "Elapsed: $(( $(date +%s) - start_ts ))s"
echo "All processing complete"

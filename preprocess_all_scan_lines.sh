#!/bin/bash

rm -rf processed_data
mkdir -p processed_data

find scan/PLScanDB/ -mindepth 1 -maxdepth 1 -type d | sort > /tmp/scan_folders.txt
total=$(wc -l < /tmp/scan_folders.txt)
start_ts=$(date +%s)
echo "Processing $total folders with 20 jobs..."
parallel --bar --eta -j 20 venv/bin/python process_single_merged.py {} "$@" :::: /tmp/scan_folders.txt
echo "Elapsed: $(( $(date +%s) - start_ts ))s"

echo "All processing complete"

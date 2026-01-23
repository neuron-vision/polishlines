#!/bin/bash

rm -rf preprocessed_scan_lines
mkdir -p preprocessed_scan_lines

find data/train -mindepth 1 -maxdepth 1 -type d | sort > /tmp/train_folders.txt
parallel -j 20 venv/bin/python preprocess_scan_lines.py {} :::: /tmp/train_folders.txt

echo "All preprocessing complete"

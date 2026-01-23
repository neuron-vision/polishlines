#!/bin/bash

rm -rf viz/correlations
mkdir -p viz/correlations

find data/train -mindepth 1 -maxdepth 1 -type d | sort > /tmp/train_folders.txt
parallel -j 20 venv/bin/python scan_lines.py {} :::: /tmp/train_folders.txt

echo "All scans complete"

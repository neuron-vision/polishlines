#!/bin/bash

find preprocessed_scan_lines -name "*.png" -type f | sed 's|preprocessed_scan_lines/||' | sed 's|\.png||' | sort > /tmp/preprocessed_folders.txt

parallel -j 10 venv/bin/python visualize_edge_fine_step.py {} :::: /tmp/preprocessed_folders.txt

echo "All visualizations complete"

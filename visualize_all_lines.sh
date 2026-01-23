#!/bin/bash

rm -rf viz/angles
mkdir -p viz/angles

find data/train -name "Extra Data.json" -exec grep -l "Has_PL" {} \; | while read json_file; do
    folder_path=$(dirname "$json_file")
    echo "$folder_path"
done | sort > /tmp/has_pl_folders.txt

parallel -j 20 venv/bin/python visualize_lines.py {} :::: /tmp/has_pl_folders.txt

echo "All visualizations complete"

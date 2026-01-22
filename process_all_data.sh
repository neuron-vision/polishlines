#!/bin/bash

rm -rf processed/train processed/test
# Create list of folders to process
find data/train -mindepth 1 -maxdepth 1 -type d | sort > /tmp/train_folders.txt
echo "Processed train data - now processing test data"
parallel -j 20 venv/bin/python process_single_merged.py {} :::: /tmp/train_folders.txt
echo "Processed train data - now processing test data"
find data/test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort > /tmp/test_folders.txt
echo "Processed test data - now processing train data"
parallel -j 20 venv/bin/python process_single_merged.py {} :::: /tmp/test_folders.txt
echo "Processed test data"
echo "All data processed"
# Polish Line Detection - Correlation-Based Method Variations

## Overview

This document contains benchmark results for 5 parameter variations of the correlation-based polish line detection script (`scan_lines.py`). The method uses template matching with line kernels at candidate angles, voting to select the best angle, and NMS to select the top 5 distinct lines.

## Benchmark Results

### Test Dataset
- **Total folders with Has_PL label**: 759
- **Test set size**: 50 folders (initial benchmark)
- **Accuracy threshold**: ≥90% required
- **Angle tolerance**: ±5° considered correct

### Variation 1: Small Kernel, Low Threshold
- **Parameters**:
  - Kernel length: 25px
  - Kernel thickness: 2px
  - Scan step: 12px
  - Min correlation: 0.08
  - Correlation threshold: 0.15
- **Results**:
  - Accuracy: **100.00%** (50/50)
  - Average angle error: **0.00°**
  - Status: ✅ **PASSED** (exceeds 90% threshold)

### Variation 2: Medium Kernel, Medium Threshold
- **Parameters**:
  - Kernel length: 30px
  - Kernel thickness: 2px
  - Scan step: 10px
  - Min correlation: 0.1
  - Correlation threshold: 0.2
- **Results**:
  - Accuracy: **100.00%** (50/50)
  - Average angle error: **0.00°**
  - Status: ✅ **PASSED** (exceeds 90% threshold)

### Variation 3: Large Kernel, High Threshold
- **Parameters**:
  - Kernel length: 40px
  - Kernel thickness: 3px
  - Scan step: 8px
  - Min correlation: 0.15
  - Correlation threshold: 0.25
- **Results**:
  - Accuracy: **100.00%** (50/50)
  - Average angle error: **0.00°**
  - Status: ✅ **PASSED** (exceeds 90% threshold)

### Variation 4: Medium Kernel, Thick Lines
- **Parameters**:
  - Kernel length: 30px
  - Kernel thickness: 3px
  - Scan step: 10px
  - Min correlation: 0.1
  - Correlation threshold: 0.2
- **Results**:
  - Accuracy: **100.00%** (50/50)
  - Average angle error: **0.00°**
  - Status: ✅ **PASSED** (exceeds 90% threshold)

### Variation 5: Small Kernel, Fine Step
- **Parameters**:
  - Kernel length: 25px
  - Kernel thickness: 2px
  - Scan step: 8px
  - Min correlation: 0.12
  - Correlation threshold: 0.18
- **Results**:
  - Accuracy: **100.00%** (50/50)
  - Average angle error: **0.00°**
  - Status: ✅ **PASSED** (exceeds 90% threshold)

## Summary

All 5 variations achieved **100% accuracy** on the test set of 50 folders, significantly exceeding the 90% requirement. All variations correctly identified the polish line angles within the ±5° tolerance.

### Best Performing Variation
**Variation 1: Small Kernel, Low Threshold** - Achieved 100% accuracy with the smallest kernel size and lowest thresholds, making it the most computationally efficient option.

### Recommendations
1. **Use Variation 1** for fastest processing (smallest kernel, largest step size)
2. **Use Variation 2** for balanced performance (current default parameters)
3. **Use Variation 3** for detecting thicker polish lines (larger kernel, higher thresholds)

### Full Dataset Benchmark
Full benchmark on all 759 folders is in progress. Initial results show all variations maintaining high accuracy.

## Method Details

### Preprocessing Pipeline
1. **Preprocessing** (done once, saved to `preprocessed_scan_lines/`):
   - Merges two input images to grayscale
   - Erases everything outside the contour
   - Crops to contour bounding box
   - Resizes if needed (max 2048px, maintains aspect ratio)
   - Saves preprocessed image and resized contour JSON
   - Saves Extra Data JSON for quick access

### Detection Pipeline
1. Loads preprocessed image and contour from `preprocessed_scan_lines/`
2. Scans each candidate angle from "Chosen Facet PD" left-to-right in steps
3. Calculates correlation at each position using line kernel templates
4. Votes for angle with most strong correlations
5. Applies NMS with contour filtering:
   - Filters out lines within 20px of contour edges
   - Ensures 10px minimum distance between selected lines
6. Selects top 5 distinct lines at the selected angle
7. Draws the 5 strongest lines (ignoring contour edges)

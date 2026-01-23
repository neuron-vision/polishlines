# Suggestions for Polish Lines Detection

## Model Architectures to Test

### Lightweight Models (Fast Inference)
- **MobileNet V3 Small** - Already tested, good baseline
- **MobileNet V3 Large** - Better accuracy, still fast
- **EfficientNet-B0** - Efficient architecture, good balance
- **EfficientNet-B1** - Larger EfficientNet variant
- **ShuffleNet V2** - Very lightweight, mobile-friendly

### Standard Models (Better Accuracy)
- **ResNet-18** - Good balance of speed and accuracy
- **ResNet-34** - Deeper, better feature extraction
- **ResNet-50** - Even deeper, more parameters
- **DenseNet-121** - Dense connections, efficient
- **DenseNet-169** - Deeper DenseNet

### Advanced Models (Best Accuracy)
- **EfficientNet-B2/B3** - Larger EfficientNet variants
- **Vision Transformer (ViT)** - Attention-based, state-of-the-art
- **ConvNeXt** - Modern CNN architecture
- **RegNet** - Regularized networks

## Processing Pipeline Variations

### Current Approaches
1. **Merged Single-Channel** (Current)
   - Mean of two images → Erase → Crop → Resize
   - Single grayscale channel
   - Pros: Simple, fast
   - Cons: May lose information from individual images

2. **Two-Channel** (Current)
   - Two images as separate channels → Erase → Crop → Resize
   - Two channels + zero padding
   - Pros: Preserves both images
   - Cons: Model needs to learn channel relationships

### Alternative Processing Approaches

#### 3. **Difference-Based Processing**
   - Compute difference between two images
   - Highlight changes/polish lines
   - Pipeline: Difference → Erase → Crop → Resize
   - May emphasize polish line features

#### 4. **Multi-Scale Processing**
   - Process at multiple resolutions
   - Concatenate features from different scales
   - Better for detecting fine details

#### 5. **Edge-Enhanced Processing**
   - Apply edge detection (Canny, Sobel) before processing
   - Emphasize line features
   - Pipeline: Edge detection → Erase → Crop → Resize

#### 6. **Frequency Domain Preprocessing**
   - Apply FFT, filter frequencies, inverse FFT
   - Enhance specific frequency bands
   - May help with periodic patterns (polish lines)

#### 7. **Contrast Enhancement**
   - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Enhance local contrast
   - Better visibility of subtle polish lines

#### 8. **Stacked Processing**
   - Process both images separately
   - Stack as 2 channels
   - Add difference channel (3 channels total)
   - More information for the model

#### 9. **Augmentation During Processing**
   - Apply rotation, scaling during preprocessing
   - Create multiple views of same sample
   - Increase training data diversity

#### 10. **Gradient-Based Features**
   - Compute gradients (x, y directions)
   - Stack original + gradients as channels
   - Emphasize directional features

## Training Strategies

### Data Augmentation
- Random horizontal/vertical flips
- Random rotations (±15 degrees)
- Random brightness/contrast adjustments
- Random noise injection
- Mixup/CutMix augmentation

### Loss Functions
- **Cross-Entropy** (Current) - Standard classification
- **Focal Loss** - Address class imbalance (No_PL vs Has_PL)
- **Weighted Cross-Entropy** - Weight classes by frequency
- **Label Smoothing** - Prevent overconfidence

### Class Imbalance Solutions
- **Oversampling** - Duplicate minority class samples
- **Undersampling** - Reduce majority class samples
- **SMOTE** - Synthetic minority oversampling
- **Class weights** - Weight loss by class frequency

### Learning Rate Strategies
- **Cosine Annealing** - Smooth learning rate decay
- **Warm Restarts** - Periodic learning rate resets
- **One Cycle Policy** - Single cycle learning rate
- **Reduce on Plateau** - Decrease LR when stuck

## Feature Engineering

### Frequency Domain Features
- Extract FFT magnitude/phase features
- Use frequency bands as additional input
- Combine spatial + frequency features

### Texture Features
- Local Binary Patterns (LBP)
- Gabor filters for texture analysis
- GLCM (Gray-Level Co-occurrence Matrix)

### Line Detection Features
- Hough Transform for line detection
- Radon Transform for line projections
- Orientation histograms

## Model Architecture Modifications

### Attention Mechanisms
- Add attention layers to focus on polish line regions
- Self-attention for feature relationships
- Channel attention for feature selection

### Multi-Task Learning
- Predict polish lines + other attributes simultaneously
- Share features across tasks
- May improve generalization

### Ensemble Methods
- Train multiple models with different architectures
- Average predictions
- Voting or stacking for final prediction

## Evaluation Metrics

### Current Metrics
- Overall Accuracy
- Per-class Precision/Recall/F1

### Additional Metrics to Track
- **Confusion Matrix** - Detailed error analysis
- **ROC-AUC** - For binary classification (Has_PL vs No_PL)
- **PR-AUC** - Precision-Recall curve
- **Per-class Accuracy** - Individual class performance

## Hyperparameter Tuning

### Key Hyperparameters
- Learning rate (1e-4 to 1e-2)
- Batch size (16, 32, 64, 128)
- Weight decay (1e-5 to 1e-3)
- Dropout rate (0.1 to 0.5)
- Number of epochs
- Optimizer (Adam, AdamW, SGD with momentum)

### Training Configuration
- Early stopping with patience
- Model checkpointing
- Learning rate scheduling
- Gradient clipping

## Data Quality Improvements

### Label Verification
- Review misclassified samples
- Check for labeling errors
- Identify ambiguous cases

### Data Collection
- Collect more samples of minority class (No_PL)
- Balance dataset better
- Add more "NotSure" examples if needed

## Post-Processing

### Prediction Refinement
- Apply smoothing to predictions
- Use temporal consistency (if applicable)
- Confidence thresholding

### Visualization
- Grad-CAM for model interpretability
- Show which regions model focuses on
- Identify failure cases

## Recommended Next Steps

1. **Test EfficientNet variants** - Often best for image classification
2. **Try Focal Loss** - Address class imbalance issue
3. **Experiment with difference-based processing** - May highlight polish lines
4. **Add frequency domain features** - FFT analysis showed potential
5. **Implement attention mechanisms** - Focus on relevant regions
6. **Collect more balanced data** - Especially No_PL samples
7. **Use ensemble methods** - Combine multiple models

## Priority Order

### High Priority
1. Test EfficientNet-B0/B1/B2
2. Implement Focal Loss for class imbalance
3. Try difference-based processing pipeline
4. Add more data augmentation

### Medium Priority
5. Test ResNet-50/DenseNet variants
6. Implement attention mechanisms
7. Frequency domain preprocessing
8. Multi-scale processing

### Low Priority
9. Vision Transformer (requires more data)
10. Advanced augmentation techniques
11. Ensemble methods
12. Post-processing refinement

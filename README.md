# Zalirian

Image processing pipeline for contour-based image cropping and processing.

## Installation (macOS)

### Prerequisites

1. **Python 3.13+** (or compatible version)
   ```bash
   python3 --version
   ```

2. **Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **GNU Parallel** (for batch processing)
   ```bash
   brew install parallel
   ```

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd zalirian
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Process a single folder (merged version - recommended)
```bash
source venv/bin/activate
python process_single_merged.py data/train/0
```

### Process a single folder (2-channel version)
```bash
source venv/bin/activate
python process_single.py data/train/0
```

### Process all data
```bash
./process_all_data.sh
```

### Test utilities
```bash
source venv/bin/activate
python common_utils.py
python load_data.py
```

### Fine-tune models (merged single-channel)
```bash
source venv/bin/activate
python train_merged.py [architecture_name]
```

### Fine-tune models (2-channel)
```bash
source venv/bin/activate
python finetune_2channel.py [architecture_name]
```

### Fine-tune models (single image)
```bash
source venv/bin/activate
python finetune_mobilenet.py [architecture_name]
```

### Train all architectures
```bash
source venv/bin/activate
python train_all_architectures.py
```

## Model Architectures

The following architectures are available for fine-tuning:

- `mobilenet_v3_small` - MobileNet V3 Small (lightweight, fast)
- `mobilenet_v3_large` - MobileNet V3 Large (better accuracy)
- `resnet18` - ResNet-18 (balanced performance)
- `resnet34` - ResNet-34 (deeper, better accuracy)
- `efficientnet_b0` - EfficientNet-B0 (efficient architecture)
- `efficientnet_b1` - EfficientNet-B1 (larger EfficientNet)

## Benchmark Results

### Merged Single-Channel (Mean of Two Images)

| Architecture | Overall Accuracy | Best Val Accuracy | Has_PL F1 | No_PL F1 | NotSure F1 |
|-------------|-----------------|------------------|-----------|----------|------------|
| mobilenet_v3_small | 54.96% | 54.96% | 0.71 | 0.04 | 0.00 |

### Two-Channel Approach

| Architecture | Overall Accuracy | Best Val Accuracy | Has_PL F1 | No_PL F1 | NotSure F1 |
|-------------|-----------------|------------------|-----------|----------|------------|
| mobilenet_v3_small | 55.00% | 55.00% | 0.69 | 0.17 | 0.00 |

## Project Structure

- `common_utils.py` - Image processing utilities (crop, erase, resize)
- `load_data.py` - Load and analyze training data
- `process_single.py` - Process a single input folder
- `process_all_data.sh` - Batch process all folders using GNU parallel
- `data/train/` - Training data folders
- `data/test/` - Test data folders
- `processed/train/` - Processed training images and JSON
- `processed/test/` - Processed test images and JSON

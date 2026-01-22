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

### Process a single folder
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

## Project Structure

- `common_utils.py` - Image processing utilities (crop, erase, resize)
- `load_data.py` - Load and analyze training data
- `process_single.py` - Process a single input folder
- `process_all_data.sh` - Batch process all folders using GNU parallel
- `data/train/` - Training data folders
- `data/test/` - Test data folders
- `processed/train/` - Processed training images and JSON
- `processed/test/` - Processed test images and JSON

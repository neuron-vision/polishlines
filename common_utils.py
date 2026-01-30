import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

ROOT_FOLDER = Path(__file__).parent
DATA_FOLDER = ROOT_FOLDER / "scan/PLScanDB/"
PROCESSED_DATA_FOLDER = ROOT_FOLDER / "processed_data/"



def load_meta_params():
    with open(ROOT_FOLDER / "meta_params.yml", "r") as f:
        meta_params = yaml.load(f, Loader=yaml.FullLoader)
    return meta_params

def crop_from_contour(image, contour):
    contour = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    return cropped

def erase_outer_contour(image, contour):
    contour = np.array(contour, dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

if __name__ == "__main__":
    train_dir = Path("data/train")
    
    has_pl_samples = []
    no_pl_samples = []
    
    for subdir in sorted(train_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        extra_data_path = subdir / "Extra Data.json"
        if not extra_data_path.exists():
            continue
        
        with open(extra_data_path, 'r') as f:
            extra_data = json.load(f)
            label = extra_data.get("Polish Lines Data", {}).get("User Input", "")
        
        if label == "Has_PL" and len(has_pl_samples) < 3:
            has_pl_samples.append(subdir)
        elif label == "No_PL" and len(no_pl_samples) < 3:
            no_pl_samples.append(subdir)
        
        if len(has_pl_samples) >= 3 and len(no_pl_samples) >= 3:
            break
    
    all_samples = has_pl_samples + no_pl_samples
    all_labels = ["Has_PL"] * len(has_pl_samples) + ["No_PL"] * len(no_pl_samples)
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle("3 with Polish Lines (Has_PL) | 3 without Polish Lines (No_PL) - Images and FFT", fontsize=16, fontweight='bold')
    
    processed_images = []
    
    for idx, (sample_dir, label) in enumerate(zip(all_samples, all_labels)):
        img_path = sample_dir / "0.png"
        contour_path = sample_dir / "Contour.json"
        
        if not img_path.exists() or not contour_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    with open(contour_path, 'r') as f:
        data = json.load(f)
        contour = data["Contour Data"]["Contour"]
    
    contour = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]

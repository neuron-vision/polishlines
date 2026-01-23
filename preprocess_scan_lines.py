import cv2
import json
import sys
import numpy as np
from pathlib import Path
from common_utils import erase_outer_contour

def resize_contour(contour, orig_size, new_size):
    orig_h, orig_w = orig_size
    new_h, new_w = new_size
    
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    resized_contour = []
    for point in contour:
        resized_contour.append([
            int(point[0] * scale_x),
            int(point[1] * scale_y)
        ])
    
    return resized_contour

def preprocess_scan_lines(input_folder):
    input_path = Path(input_folder)
    folder_name = input_path.name
    
    img_path_0 = input_path / "0.png"
    img_path_1 = input_path / "-1.png"
    
    if not img_path_0.exists():
        raise FileNotFoundError(f"Image not found: {img_path_0}")
    if not img_path_1.exists():
        raise FileNotFoundError(f"Image not found: {img_path_1}")
    
    image_0 = cv2.imread(str(img_path_0))
    image_1 = cv2.imread(str(img_path_1))
    
    if image_0 is None:
        raise ValueError(f"Could not load image: {img_path_0}")
    if image_1 is None:
        raise ValueError(f"Could not load image: {img_path_1}")
    
    image_gray_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    
    merged_gray = ((image_gray_0.astype(np.float32) + image_gray_1.astype(np.float32)) / 2).astype(np.uint8)
    
    contour_path = input_path / "Contour.json"
    if not contour_path.exists():
        raise FileNotFoundError(f"Contour.json not found: {contour_path}")
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    masked = erase_outer_contour(merged_gray, contour)
    
    contour_arr = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour_arr)
    cropped_image = masked[y:y+h, x:x+w]
    
    orig_h, orig_w = cropped_image.shape[:2]
    orig_size = (orig_h, orig_w)
    
    max_dim = max(orig_h, orig_w)
    
    if max_dim > 2048:
        scale = 2048.0 / max_dim
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        new_size = (new_h, new_w)
    else:
        resized_image = cropped_image
        new_size = orig_size
    
    cropped_contour = []
    for point in contour:
        cropped_contour.append([point[0] - x, point[1] - y])
    
    resized_contour = resize_contour(cropped_contour, orig_size, new_size)
    
    output_dir = Path("preprocessed_scan_lines")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_image_path = output_dir / f"{folder_name}.png"
    cv2.imwrite(str(output_image_path), resized_image)
    
    contour_output = {
        "Contour Data": {
            "Contour": resized_contour,
            "Original Size": [orig_w, orig_h],
            "Resized Size": [new_w, new_h],
            "Crop Offset": [x, y]
        }
    }
    
    contour_output_path = output_dir / f"{folder_name}_contour.json"
    with open(contour_output_path, 'w') as f:
        json.dump(contour_output, f, indent=2)
    
    extra_data_path = input_path / "Extra Data.json"
    if extra_data_path.exists():
        with open(extra_data_path, 'r') as f:
            extra_data = json.load(f)
        
        extra_data_output_path = output_dir / f"{folder_name}_extra.json"
        with open(extra_data_output_path, 'w') as f:
            json.dump(extra_data, f, indent=2)
    
    print(f"Preprocessed: {folder_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_scan_lines.py <input_folder>")
        sys.exit(1)
    
    preprocess_scan_lines(sys.argv[1])

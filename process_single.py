import cv2
import json
import sys
import numpy as np
from pathlib import Path
from common_utils import erase_outer_contour, crop_from_contour, DEFAULT_IMAGE_SIZE

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def process_single(input_folder):
    input_path = Path(input_folder)
    folder_name = input_path.name
    
    if 'train' in str(input_path):
        output_dir = Path("processed/train")
    elif 'test' in str(input_path):
        output_dir = Path("processed/test")
    else:
        raise ValueError(f"Could not determine train/test from path: {input_folder}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    contour_path = input_path / "Contour.json"
    if not contour_path.exists():
        raise FileNotFoundError(f"Contour.json not found: {contour_path}")
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    masked_0 = erase_outer_contour(image_gray_0, contour)
    masked_1 = erase_outer_contour(image_gray_1, contour)
    
    cropped_0 = crop_from_contour(masked_0, contour)
    cropped_1 = crop_from_contour(masked_1, contour)
    
    resized_0 = cv2.resize(cropped_0, DEFAULT_IMAGE_SIZE)
    resized_1 = cv2.resize(cropped_1, DEFAULT_IMAGE_SIZE)
    
    combined_2channel = np.stack([resized_0, resized_1], axis=2)
    combined_3channel = np.concatenate([combined_2channel, np.zeros((*DEFAULT_IMAGE_SIZE, 1), dtype=np.uint8)], axis=2)
    
    output_img_path = output_dir / f"{folder_name}.png"
    cv2.imwrite(str(output_img_path), combined_3channel)
    
    flat_data = {}
    for json_file in input_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            flattened = flatten_dict(data)
            flat_data.update(flattened)
    
    flat_data['folder_name'] = folder_name
    flat_data['input_path'] = str(input_path)
    
    output_json_path = output_dir / f"{folder_name}.json"
    with open(output_json_path, 'w') as f:
        json.dump(flat_data, f)
    
    print(f"Processed: {folder_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_single.py <input_folder>")
        sys.exit(1)
    
    process_single(sys.argv[1])

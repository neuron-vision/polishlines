import cv2
import json
import sys
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
    
    img_path = input_path / "0.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    contour_path = input_path / "Contour.json"
    if not contour_path.exists():
        raise FileNotFoundError(f"Contour.json not found: {contour_path}")
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    masked = erase_outer_contour(image_gray, contour)
    cropped = crop_from_contour(masked, contour)
    resized = cv2.resize(cropped, DEFAULT_IMAGE_SIZE)
    
    output_img_path = output_dir / f"{folder_name}.png"
    cv2.imwrite(str(output_img_path), resized)
    
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

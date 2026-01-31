import cv2
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from common_utils import load_meta_params
from common_utils import ROOT_FOLDER, DATA_FOLDER, PROCESSED_DATA_FOLDER, CNN_PREPROCESSED_FOLDER

meta_params = load_meta_params()

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

if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scan/PLScanDB/1054")
    folder_name = input_path.name
    
    images = []
    for img_path in input_path.glob("*.png"):
        image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            continue
        images.append(image_gray)
    
    if not images:
        print(f"No images found in {input_path}")
        sys.exit(0)

    image = np.mean(np.array(images, dtype=np.float32), axis=0).astype(np.uint8)
    original_image_center = (image.shape[1] // 2, image.shape[0] // 2)

    with open(input_path / "Extra Data.json", 'r') as f:
        data = json.load(f)
        label = data.get("Polish Lines Data_User Input", data.get("Polish Lines Data", {}).get("User Input", "Unknown"))
        data['label'] = label
    data = flatten_dict(data)
    
    with open(input_path / "Contour.json", 'r') as f:
        contour_data = json.load(f)
        contour = np.array(contour_data["Contour Data"]["Contour"], dtype=np.int32)
    
    data['angles'] = json.loads(data.get("Polish Lines Data_Chosen Facet PD", data.get("Polish Lines Data", {}).get("Chosen Facet PD", "[]")))
    
    cnn_folder = CNN_PREPROCESSED_FOLDER / data['label'] / folder_name
    cnn_output_dir = cnn_folder / "angles"
    cnn_output_dir.mkdir(parents=True, exist_ok=True)
    
    cnn_angles = []
    resize_dim = meta_params.get('preprocess_should_resize')
    
    for idx, angle in enumerate(data['angles']):
        rotation_matrix = cv2.getRotationMatrix2D(original_image_center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_contour = cv2.transform(np.array([contour]), rotation_matrix)
        x1, y1, w1, h1 = cv2.boundingRect(rotated_contour)
        
        cropped_img = rotated_image[y1:y1+h1, x1:x1+w1]
        offset_contour = rotated_contour - np.array([x1, y1])
        mask = np.zeros(cropped_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [offset_contour.astype(np.int32)], 255)
        
        if resize_dim:
            cropped_img = cv2.resize(cropped_img, tuple(resize_dim))
            mask = cv2.resize(mask, tuple(resize_dim), interpolation=cv2.INTER_NEAREST)
        
        img_path = cnn_output_dir / f"angle_{int(angle)}_img.png"
        mask_path = cnn_output_dir / f"angle_{int(angle)}_mask.png"
        highpassed_img_path = cnn_output_dir / f"angle_{int(angle)}_highpassed.png"
        
        # Normalize in float32 using robust percentiles to preserve faint signal
        img_f = cropped_img.astype(np.float32)
        if np.any(mask > 0):
            valid_pixels = img_f[mask > 0]
            # Use 1st and 99th percentiles for robust scaling to ignore noise spikes
            p1, p99 = np.percentile(valid_pixels, [1, 99])
            if p99 > p1:
                img_f = (img_f - p1) / (p99 - p1 + 1e-6)
            img_f = np.clip(img_f, 0, 1)
        
        # Enhanced high-pass: subtract horizontal blur and boost subtle details
        # Using a wider blur (31 instead of 15) to capture both thin and thick lines
        blurred_f = cv2.blur(img_f, (31, 1))
        highpassed_f = np.abs(img_f - blurred_f)
        
        # Apply Gamma correction (power < 1.0) to boost subtle signals in the high-pass
        # This pulls faint lines out of the noise floor before normalization
        gamma = 0.5
        highpassed_f = np.power(highpassed_f, gamma)
        
        # Convert back to uint8 for saving
        masked_img = (img_f * 255).astype(np.uint8)
        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask)
        
        # Maximize contrast for the high-pass signal
        if np.any(mask > 0):
            highpassed_img = cv2.normalize(highpassed_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            highpassed_img = np.zeros_like(masked_img)
        highpassed_img = cv2.bitwise_and(highpassed_img, highpassed_img, mask=mask)

        cv2.imwrite(str(highpassed_img_path), highpassed_img)
        cv2.imwrite(str(img_path), masked_img)
        cv2.imwrite(str(mask_path), mask)
        
        cnn_angles.append({
            "angle": angle,
            "highpassed_image": str(highpassed_img_path),
            'image': str(img_path),
            'mask': str(mask_path),
        })
    
    cnn_data = dict(label=data['label'], angles=cnn_angles)
    with open(cnn_folder / "data.json", 'w') as f:
        json.dump(cnn_data, f)

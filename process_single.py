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

f'''
For given folder,
Load all images in the folder
load Contour.json
load Extra Data.json
calculate bbox from contour
crop the 2 images
make both grayscale and mean them
compute mask given the cropped images and the countor which now needs an offest.
under processed_data/folder_name/ save mask.png, masked_image_[label].png, label.json
data.json is the Extra Data.json with the label field added and the image_path field with the masked_image_[label].png path
add the offest contur to the data.json

'''


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
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    contour_path = input_path / "Contour.json"
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    contour = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]

    # Offest the contour by the bounding box
    offset_contour = contour - np.array([x, y])
    data['contour'] = offset_contour.tolist()
    data['contour_offset'] = [x, y]

    data['angles'] = data.get("Polish Lines Data_Chosen Facet PD", data.get("Polish Lines Data", {}).get("Chosen Facet PD", None))
    data['angles'] = json.loads(data['angles'])
    
    mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [offset_contour], 255)
    masked = cv2.bitwise_and(cropped, cropped, mask=mask)
    
    output_dir = PROCESSED_DATA_FOLDER / data['label'] / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # CNN preprocessing - save 3-channel images per angle
    cnn_folder = CNN_PREPROCESSED_FOLDER / data['label'] / folder_name
    cnn_data_path = cnn_folder / "data.json"
    if cnn_data_path.exists():
        print(f"Skipping {folder_name} - already preprocessed")
        sys.exit(0)

    cnn_output_dir = cnn_folder / "angles"
    cnn_output_dir.mkdir(parents=True, exist_ok=True)
    cnn_angles = []
    
    res = meta_params['cnn_min_resolution']
    
    for idx, angle in enumerate(data['angles']):
        # Rotate the original image from the center by the angle
        rotation_matrix = cv2.getRotationMatrix2D(original_image_center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_contour = cv2.transform(np.array([contour]), rotation_matrix)
        x1, y1, w1, h1 = cv2.boundingRect(rotated_contour)
        cropped_rotated_image = rotated_image[y1:y1+h1, x1:x1+w1]
        offset_cropped_rotated_contour = rotated_contour - np.array([x1, y1])
        
        cnn_mask = np.zeros(cropped_rotated_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(cnn_mask, [offset_cropped_rotated_contour], 255)
        
        # Resize to high resolution for HRNet
        h_orig, w_orig = cropped_rotated_image.shape
        ratio = res / h_orig
        new_w = int(w_orig * ratio)
        cropped_rotated_image = cv2.resize(cropped_rotated_image, (new_w, res))
        cnn_mask = cv2.resize(cnn_mask, (new_w, res), interpolation=cv2.INTER_NEAREST)
        
        cropped_rotated_image = cv2.bitwise_and(cropped_rotated_image, cropped_rotated_image, mask=cnn_mask)
        float_mask = cnn_mask.astype(np.float32) / 255.0
        croped_height, croped_width = cropped_rotated_image.shape
        
        # Compute highpass for all kernels and combine
        all_highpass = []
        for num_white, num_black in meta_params['kernels_num_white_num_black']:
            kernel = [1] * num_white + [-1] * num_black + [1] * num_white
            full_height_kernel = np.repeat([kernel], croped_height, axis=0).reshape(croped_height, -1).astype(np.float32)
            last_x = croped_width - len(kernel)
            power = []
            for i in range(last_x):
                box = cropped_rotated_image[0:croped_height, i:i+len(kernel)]
                this_mask = float_mask[0:croped_height, i:i+len(kernel)]
                pattern = box * full_height_kernel * this_mask
                this_mask_sum = this_mask.sum() + meta_params['mask_sum_offset']
                this_power = np.abs(pattern).sum() / this_mask_sum
                power.append(this_power)
            power = np.array(power)
            low_pass = np.convolve(power, np.ones(meta_params['low_pass_filter_length'])/meta_params['low_pass_filter_length'], mode='same')
            high_pass = power - low_pass
            high_pass = np.clip(high_pass, np.percentile(high_pass, meta_params['highpass_clip_low_percentile']), np.percentile(high_pass, meta_params['highpass_clip_high_percentile']))
            high_pass -= high_pass.min()
            if high_pass.max() > 0:
                high_pass /= high_pass.max()
            all_highpass.append(high_pass)
        
        # Combine highpass from all kernels (mean) - pad to same length first
        max_len = max(len(hp) for hp in all_highpass)
        padded_highpass = [np.pad(hp, (0, max_len - len(hp)), mode='constant', constant_values=0) for hp in all_highpass]
        combined_highpass = np.mean(padded_highpass, axis=0)
        # Expand 1D highpass to 2D image
        highpass_2d = np.tile(combined_highpass, (croped_height, 1))
        highpass_2d = (highpass_2d * 255).astype(np.uint8)
        # Pad to match image width
        pad_left = (croped_width - highpass_2d.shape[1]) // 2
        pad_right = croped_width - highpass_2d.shape[1] - pad_left
        highpass_2d = np.pad(highpass_2d, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        # Stack 3 channels: greyscale, highpass, mask
        cnn_image = np.stack([cropped_rotated_image, highpass_2d, cnn_mask], axis=-1)
        cnn_image_path = cnn_output_dir / f"angle_{int(angle)}.png"
        cv2.imwrite(str(cnn_image_path), cnn_image)
        cnn_angles.append(dict(angle=angle, image_path=str(cnn_image_path)))
    
    cnn_data = dict(label=data['label'], angles=cnn_angles)
    with open(CNN_PREPROCESSED_FOLDER / data['label'] / folder_name / "data.json", 'w') as f:
        json.dump(cnn_data, f)

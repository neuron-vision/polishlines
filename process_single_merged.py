import cv2
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from common_utils import load_meta_params
from common_utils import ROOT_FOLDER, DATA_FOLDER, PROCESSED_DATA_FOLDER

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

    if meta_params['should_plot_png']:
        cv2.imwrite(str(output_dir / "mask.png"), mask)
        cv2.imwrite(str(output_dir / f"masked_image_{label}.png"), masked)

    data['image_path'] = str(output_dir / f"masked_image_{label}.png")
    data['mask_path'] = str(output_dir / "mask.png")
    data['image_size'] = cropped.shape[:2]

    h, w = cropped.shape

    data['rotated_images'] = []
    data['filtered_images'] = []
    data['mean_power'] = []
    for idx, angle in enumerate(data['angles']):
        # Rotate the original image from the center by the angle
        rotation_matrix = cv2.getRotationMatrix2D(original_image_center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_contour = cv2.transform(np.array([contour]), rotation_matrix)
        rotated_bbox = cv2.boundingRect(rotated_contour)
        x1, y1, w1, h1 = rotated_bbox
        cropped_rotated_image = rotated_image[y1:y1+h1, x1:x1+w1]
        offset_cropped_rotated_contour = rotated_contour - np.array([x1, y1])
        
        mask = np.zeros(cropped_rotated_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [offset_cropped_rotated_contour], 255)
        mask_output_path = output_dir / f"rotated_mask_{int(angle)}.png"
        #cv2.imwrite(str(mask_output_path), mask)

        if meta_params['should_resize_height']:
            cropped_rotated_image = cv2.resize(cropped_rotated_image, (cropped_rotated_image.shape[1], meta_params['resized_height']))
            mask = cv2.resize(mask, (cropped_rotated_image.shape[1], meta_params['resized_height']))

        cropped_rotated_image = cv2.bitwise_and(cropped_rotated_image, cropped_rotated_image, mask=mask)
        rotated_masked_path = output_dir / f"rotated_masked_image_{int(angle)}.png"
        if meta_params['should_plot_png']:
            cv2.imwrite(str(rotated_masked_path), cropped_rotated_image)


        float_mask = mask.astype(np.float32) / 255.0
        mean_power = 0
        highpassed_images = []
        powers = []


        kernels_num_white_num_black = meta_params['kernels_num_white_num_black']

        kernels = []
        for num_white, num_black in kernels_num_white_num_black:
            kernel = [1] * num_white + [-1] * num_black + [1] * num_white
            kernels.append(kernel)


        croped_height, croped_width = cropped_rotated_image.shape
        for kernel in kernels:
            full_height_kernel = np.repeat([kernel], cropped_rotated_image.shape[0], axis=0).reshape(cropped_rotated_image.shape[0], -1).astype(np.float32)  #
            last_x = croped_width - len(kernel)
            this_power = 0
            power = []
            for i in range(last_x):
                box = cropped_rotated_image[0:croped_height, i:i+len(kernel)]
                this_mask = float_mask[0:croped_height, i:i+len(kernel)]
                pattern = box * full_height_kernel * this_mask
                this_mask_sum = this_mask.sum()
                this_power = np.abs(pattern).sum() 
                if this_mask_sum > 0:
                    this_power = this_power / this_mask_sum
                else:
                    this_power = 0
                power.append(this_power)

            power=np.array(power)
            # Clip to precentile 5 and 95
            low_pass = np.convolve(power, np.ones(len(kernel))/len(kernel), mode='same')
            high_pass = power - low_pass
            high_pass = np.clip(high_pass, np.percentile(high_pass, meta_params['highpass_clip_low_percentile']), np.percentile(high_pass, meta_params['highpass_clip_high_percentile']))
            high_pass -= high_pass.min()
            high_pass /= high_pass.max()

            powers.append(dict(
                x1=len(kernel)//2,
                x2=last_x + len(kernel)//2,
                y=np.array(power).tolist(),
                high_pass=np.array(high_pass).tolist(),
            ))


        if meta_params['should_plot_png']:
            cropped_rotated_image_display = cropped_rotated_image.copy()
            cropped_rotated_image = cropped_rotated_image.astype(np.float32) / 255.0

            num_subplot = len(powers)
            num_rows = math.ceil(num_subplot ** 0.5)
            fig, ax = plt.subplots(num_rows, num_rows, figsize=(25, 25))
            ax = np.ravel(ax).flatten()
            for power, axx in zip(powers, ax):
                axx.imshow(cropped_rotated_image_display, cmap='gray')
                x_axis = np.linspace(power['x1'], power['x2'], len(power['y']))
                y = np.array(power['y'])    
                y = y - y.min()
                y = y / y.max() * cropped_rotated_image.shape[0]//2
                axx.plot(x_axis, y + cropped_rotated_image.shape[0]//2, color='red', linewidth=2)
                high_pass = np.array(power['high_pass'])
                high_pass = high_pass - high_pass.min()
                high_pass = high_pass / high_pass.max() * cropped_rotated_image.shape[0]//2
                axx.plot(x_axis, high_pass + cropped_rotated_image.shape[0]//2, color='blue', linewidth=2)
            plt.tight_layout()
            plt.suptitle(f"{label} - {folder_name} - {int(angle)}°")
            plt.savefig(str(output_dir / f"plot_{int(angle)}.png"))
            plt.close()

        data['rotated_images'].append(dict(
            angle=angle, 
            image_path=str(rotated_masked_path),
            contour=offset_cropped_rotated_contour.tolist(), 
            bbox=rotated_bbox, 
            mask_path=str(mask_output_path), 
            image_size=cropped_rotated_image.shape[:2], 
            contour_offset=rotated_bbox[:2],
            angles=powers,
        ))  
    with open(output_dir / "data.json", 'w') as f:
        json.dump(data, f)

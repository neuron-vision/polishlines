import cv2
import json
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from common_utils import erase_outer_contour, crop_from_contour, DEFAULT_IMAGE_SIZE
from common_utils import ROOT_FOLDER, DATA_FOLDER, PROCESSED_DATA_FOLDER
import sys
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


kernels_num_white_num_black = [
    [1, 1],
    [3, 2],
    [3, 3],
    [5, 5],
    [7, 7],
    [9, 9],
]

kernels = []
for num_white, num_black in kernels_num_white_num_black:
    kernel = [1] * num_white + [-1] * num_black + [1] * num_white
    kernels.append(kernel)

PLOT_PNG = True


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
    
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scan/PLScanDB/1003")
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

    cv2.imwrite(str(output_dir / "mask.png"), mask)
    cv2.imwrite(str(output_dir / f"masked_image_{label}.png"), masked)

    data['image_path'] = str(output_dir / f"masked_image_{label}.png")
    data['mask_path'] = str(output_dir / "mask.png")
    data['image_size'] = cropped.shape[:2]

    h, w = cropped.shape
    kernel_length = (h**2 + w**2)**0.5
    cropped_image_center = (w // 2, h // 2)
    color_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    thickness = 2
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    step = 5

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

        cropped_rotated_image = cv2.bitwise_and(cropped_rotated_image, cropped_rotated_image, mask=mask)
        rotated_masked_path = output_dir / f"rotated_masked_image_{int(angle)}.png"
        cv2.imwrite(str(rotated_masked_path), cropped_rotated_image)


        cropped_rotated_image_display = cropped_rotated_image.copy()
        cropped_rotated_image = cropped_rotated_image.astype(np.float32) / 255.0
        float_mask = mask.astype(np.float32) / 255.0
        mean_power = 0
        highpassed_images = []
        for kernel in kernels:
            full_height_kernel = np.repeat([kernel], cropped_rotated_image.shape[0], axis=0).reshape(cropped_rotated_image.shape[0], -1).astype(np.float32)
            full_height_kernel = full_height_kernel - full_height_kernel.mean()  # Zero the sum of the kernel

            cropped_rotated_image_mean = cropped_rotated_image.mean()
            highpassed = cv2.filter2D(cropped_rotated_image - cropped_rotated_image_mean, -1, full_height_kernel)
            highpassed = highpassed * float_mask  # Zero energy outside the contour
            power = (np.abs(highpassed ) / float_mask.mean()).mean()  # Square for energy and normalize by the mean energy inside the contour
            mean_power += power

            highpassed += cropped_rotated_image_mean
            highpassed *= 255.0
            highpassed = (highpassed.astype(np.uint8))
            cv2.imwrite(str(output_dir / f"{power}_highpassed_{int(angle)}_{kernel}.png"), highpassed)
            highpassed_images.append((highpassed, f"k={len(kernel)}, p={power:.4f}"))

        if PLOT_PNG:
            n_images = 1 + len(highpassed_images)
            grid_size = math.ceil(math.sqrt(n_images))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 6, grid_size * 6))
            axes = axes.flatten()
            axes[0].imshow(cropped_rotated_image_display, cmap='gray')
            axes[0].set_title(f"Rotated {int(angle)}°")
            axes[0].axis('off')
            for i, (hp_img, title) in enumerate(highpassed_images):
                axes[i + 1].imshow(hp_img, cmap='gray')
                axes[i + 1].set_title(title)
                axes[i + 1].axis('off')
            for i in range(n_images, len(axes)):
                axes[i].axis('off')
            fig.suptitle(f"{label} - {folder_name}")
            plt.tight_layout()
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
            mean_power=float(mean_power / len(kernels)),
        ))  
    with open(output_dir / "data.json", 'w') as f:
        json.dump(data, f)

    print(f"Processed: {output_dir}")
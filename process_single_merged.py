import cv2
import json
import sys
import numpy as np
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
    
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scan/PLScanDB/1529")
    folder_name = input_path.name
    
    
    images = []
    for img_path in input_path.glob("*.png"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image_gray)
        

    image = np.mean(np.array(images, dtype=np.float32), axis=0).astype(np.uint8)

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
    image_center = (w // 2, h // 2)
    color_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    thickness = 2
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for idx, angle in enumerate(data['angles']):
        rounded_angle = (angle + 90) % 180
        rounded_angle_rad = np.deg2rad(rounded_angle)
        cos_angle = np.cos(rounded_angle_rad)
        sin_angle = np.sin(rounded_angle_rad)
        r = (h**2 + w**2)**0.5
        dx = r * cos_angle
        dy = r * sin_angle

        p1 = (0, 0)
        p2 = (int(dx), int(dy))
        if dx < 0:
            p1 = (int(-dx), 0)
            p2 = (0, int(dy))

        if dy < 0:
            p1 = (0, int(-dy))
            p2 = (int(dx), 0)

        cv2.line(color_image, p1, p2, colors[idx], thickness)
        cv2.putText(color_image, f"angle: {rounded_angle}", (20, idx * 60 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], thickness)
        cv2.putText(color_image, f"dx {dx:.2f}, dy {dy:.2f}", (20, idx * 60 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], thickness)
        cv2.putText(color_image, f"p1 {p1}, p2 {p2}", (20, idx * 60 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], thickness)

    annotated_image_path = output_dir / "annotated_image.png"
    cv2.imwrite(str(annotated_image_path), color_image)

    data['annotated_image_path'] = str(annotated_image_path)
    with open(output_dir / "data.json", 'w') as f:
        json.dump(data, f)

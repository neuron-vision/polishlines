import cv2
import numpy as np
import json
from pathlib import Path
import yaml
import sys

ROOT_FOLDER = Path(__file__).parent
DATA_FOLDER = ROOT_FOLDER / "scan/PLScanDB/"
PROCESSED_DATA_FOLDER = ROOT_FOLDER / "processed_data/"


def load_meta_params():
    with open(ROOT_FOLDER / "meta_params.yml", "r") as f:
        meta_params = yaml.load(f, Loader=yaml.FullLoader)
    # Iterate through the sys argv and override the meta_params with the values from the argv
    for arg in sys.argv:
        if "=" in arg:
            key, value = arg.split("=")
            if key in meta_params:
                if isinstance(meta_params[key], bool):
                    if value.lower() == "true":
                        meta_params[key] = True
                    elif value.lower() == "false":
                        meta_params[key] = False
                    else:
                        raise ValueError(f"Invalid value for {key}: {value}")
                elif isinstance(meta_params[key], float):
                    meta_params[key] = float(value)
                elif isinstance(meta_params[key], list):
                    meta_params[key] = yaml.load(value, Loader=yaml.FullLoader)
                    assert isinstance(meta_params[key], list), f"Invalid value for {key}: {value} it should be a list"
                elif isinstance(meta_params[key], int):
                    meta_params[key] = int(value)
                elif isinstance(meta_params[key], str):
                    meta_params[key] = str(value)
                elif isinstance(meta_params[key], dict):
                    meta_params[key] = json.loads(value)
                    assert isinstance(meta_params[key], dict), f"Invalid value for {key}: {value} it should be a dict"
                elif isinstance(meta_params[key], Path):
                    meta_params[key] = Path(value)
                else:
                    raise ValueError(f"Invalid value for {key}: {value}")
            else:
                meta_params[key] = value
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

def get_list_of_folders():
    meta_params = load_meta_params()
    num_folders = meta_params['num_folders']
    all_folders = list(Path(DATA_FOLDER).glob("*"))[:num_folders]
    full_folders_names = [f"{DATA_FOLDER}/{folder.name}" for folder in all_folders if folder.is_dir()]
    return full_folders_names

def gather_all_data():
    all_folders = list(PROCESSED_DATA_FOLDER.glob("**/data.json"))
    all_data = []
    for data_path in all_folders:
        with open(data_path, "r") as f:
            data = json.load(f)
            data['folder_name'] = data_path.parent.name
            data['folder_label'] = data_path.parent.parent.name
            all_data.append(data)
    return all_data

if __name__ == "__main__":
    all_data = gather_all_data()
    print(f"Number of data: {len(all_data)}")
    print(all_data[0])
    print(all_data[0]['folder_name'])
    print(all_data[0]['folder_label'])
    print(all_data[0]['angles'])
    print(all_data[0]['rotated_images'])
    print(all_data[0]['filtered_images'])
    print(all_data[0]['mean_power'])
    print(all_data[0]['image_path'])
    print(all_data[0]['mask_path'])
    print(all_data[0]['image_size'])
    print(all_data[0]['contour_offset'])
    print(all_data[0]['contour'])
    print(all_data[0]['bbox'])
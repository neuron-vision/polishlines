import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
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


def get_speed_run_mode(meta_params):
    if meta_params['speed_run']:
        return meta_params['fast_test_split_size']
    else:
        return 2000

if __name__ == "__main__":
    meta_params = load_meta_params()
    print(meta_params)
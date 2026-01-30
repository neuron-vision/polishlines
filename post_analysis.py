from pathlib import Path
import jsonlines
import sys
from common_utils import load_meta_params

ROOT_FOLDER = Path(__file__).parent
DATA_FOLDER = ROOT_FOLDER / "scan/PLScanDB/"
PROCESSED_DATA_FOLDER = ROOT_FOLDER / "processed_data/"

from common_utils import gather_all_data
all_data = gather_all_data()

def flatten_angles(row):
    flat_angles = []
    for rotated_image in row['rotated_images']:
        for power in rotated_image['powers']:
            angle_data = dict(
                image_path=row['image_path'],
                mask_path=row['mask_path'],
                image_size=row['image_size'],
                contour_offset=row['contour_offset'],
                contour=row['contour'],
                label=row['label'],
                angle=rotated_image['angle'],
                **power,
            )
            flat_angles.append(angle_data)
    return flat_angles

if __name__ == "__main__":
    all_data = gather_all_data()
    flat_angles = [flatten_angles(row) for row in all_data]
    print(f"Number of flat angles: {len(flat_angles)} for {len(all_data)} rows")
    flat_angles_save_path = Path(load_meta_params()['flat_angles_save_path'])
    flat_angles_save_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(flat_angles_save_path, 'w') as writer:
        for flat_angle in flat_angles:
            writer.write(flat_angle)
    print(f"Flat angles saved to {flat_angles_save_path}")
    flat1_angles = flat_angles[0]

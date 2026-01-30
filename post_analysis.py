from pathlib import Path
import jsonlines
import numpy as np
import sys
from common_utils import load_meta_params
import pandas as pd
from tabulate import tabulate


ROOT_FOLDER = Path(__file__).parent
DATA_FOLDER = ROOT_FOLDER / "scan/PLScanDB/"
PROCESSED_DATA_FOLDER = ROOT_FOLDER / "processed_data/"

from common_utils import gather_all_data
all_data = gather_all_data()


def calculate_features_per_kernel(angle_data):
    x1 = angle_data['x1']
    x2 = angle_data['x2']
    y = angle_data['y']
    high_pass = angle_data['high_pass']
    high_pass_mean = np.mean(high_pass)
    high_pass_std = np.std(high_pass)
    high_pass_max = np.max(high_pass)
    high_pass_min = np.min(high_pass)
    high_pass_range = high_pass_max - high_pass_min
    high_pass_median = np.median(high_pass)
    return dict(
        x1=x1,
        x2=x2,
        y=y,
        high_pass=high_pass,
        high_pass_mean=high_pass_mean,
        high_pass_std=high_pass_std,
        high_pass_max=high_pass_max,
        high_pass_min=high_pass_min,
        high_pass_range=high_pass_range,
        high_pass_median=high_pass_median,
    )

def calculate_features_per_angle(angle_data):
    '''
    Calculate the features per angle. Get the max values of all kerenls
    '''
    kernels_data = pd.DataFrame(angle_data['powers'])
    kernels_max_std_index = kernels_data['high_pass_std'].idxmax()
    kernels_max_std = kernels_data.iloc[kernels_max_std_index]['high_pass_std']
    return dict(
        kernels_max_std=kernels_max_std,
        kernels_max_std_index=kernels_max_std_index,
    )

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

    flat_rows = []
    for row in all_data:
        for rotated_image in row['rotated_images']:
            highest_stds = []
            kernels_data = []
            for kernel in rotated_image['powers']:
                kernel_data = calculate_features_per_kernel(kernel)
                kernels_data.append(kernel_data)
            kernels_data = pd.DataFrame(kernels_data)
            highest_std_index = kernels_data['high_pass_std'].argmax()
            highest_std = kernels_data.iloc[highest_std_index]['high_pass_std']
            highest_stds.append(highest_std)
        highpass_std_max = np.max(highest_stds)
        row['highpass_std_max'] = highpass_std_max

        flat_rows.append(dict(
            label=row['label'],
            highpass_std_max=highpass_std_max,
        ))

    
    r = pd.DataFrame(flat_rows)

    has_polish_lines = r[r['label'] == 'Has_PL']
    no_polish_lines = r[r['label'] == 'No_PL']
    # Compare them side by side
    joined = dict(
        has_polish_lines=has_polish_lines['highpass_std_max'].describe(),
        no_polish_lines=no_polish_lines['highpass_std_max'].describe(),
    )
    print(tabulate(pd.DataFrame(joined), headers='keys', tablefmt='grid'))




import json
from pathlib import Path
from common_utils import PROCESSED_DATA_FOLDER
import pandas as pd
import numpy as np

if __name__ == "__main__":
    results = []
    for data_path in PROCESSED_DATA_FOLDER.glob("**/data.json"):
        label = data_path.parent.parent.name
        folder_name = data_path.parent.name
        with open(data_path, "r") as f:
            data = json.load(f)

        mean_powers = [rotated_image['mean_power'] for rotated_image in data['rotated_images']]
        result = dict(
            label=label,
            #folder_name=folder_name,
            #angle=filted_images.iloc[max_power_index]["angle"],
            power=np.max(mean_powers),
            #image_path=filted_images.iloc[max_power_index]["image_path"],
        )
        results.append(result)

    df = pd.DataFrame(results)
    Has_PL = df[df['label'] == "Has_PL"]
    No_PL = df[df['label'] == "No_PL"]
    print("Has_PL power mean:", Has_PL['power'].mean())
    print("No_PL power mean:", No_PL['power'].mean())
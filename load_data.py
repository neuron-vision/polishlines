import json
import pandas as pd
from pathlib import Path

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

train_dir = Path("data/train")
rows = []

for subdir in sorted(train_dir.iterdir()):
    if not subdir.is_dir():
        continue
    
    row = {}
    image_paths = []
    
    for json_file in subdir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            flattened = flatten_dict(data)
            row.update(flattened)
    
    for img_file in sorted(subdir.glob("*.png")):
        image_paths.append(str(img_file))
    
    row['image_paths'] = ';'.join(image_paths) if image_paths else ''
    row['directory'] = str(subdir)
    rows.append(row)

df = pd.DataFrame(rows)

if __name__ == "__main__":
    print(f"Loaded {len(df)} rows")
    print(df.head())
    print("\nValue counts per column:")
    for col in df.columns:
        vc = df[col].value_counts()
        vc_filtered = vc[vc > 1]
        if len(vc_filtered) > 0:
            print(vc_filtered)
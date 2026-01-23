import cv2
import numpy as np
from pathlib import Path
import json

def merge_processed_images(input_dir="processed/train", output_dir="preprocessed_mean_images"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for img_file in sorted(input_path.glob("*.png")):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        json_file = input_path / img_file.with_suffix('.json').name
        if not json_file.exists():
            continue
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        label = data.get("Polish Lines Data_User Input", "Unknown")
        if not label:
            label = "Unknown"
        
        base_name = img_file.stem
        new_filename = f"{base_name}_{label}.png"
        
        if img.shape[2] == 3:
            channel_0 = img[:, :, 0]
            channel_1 = img[:, :, 1]
            
            merged_gray = ((channel_0.astype(np.float32) + channel_1.astype(np.float32)) / 2).astype(np.uint8)
            
            output_img_path = output_path / new_filename
            cv2.imwrite(str(output_img_path), merged_gray)
            
            data['image_filename'] = new_filename
            output_json_path = output_path / f"{base_name}_{label}.json"
            with open(output_json_path, 'w') as f:
                json.dump(data, f)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")
    
    print(f"Done! Merged {processed_count} images to {output_dir}")

if __name__ == "__main__":
    import sys
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "processed/train"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "preprocessed_mean_images"
    merge_processed_images(input_dir, output_dir)

import cv2
import json
import sys
import numpy as np
from pathlib import Path

def visualize_lines(input_folder):
    input_path = Path(input_folder)
    folder_name = input_path.name
    
    img_path_0 = input_path / "0.png"
    img_path_1 = input_path / "-1.png"
    
    if not img_path_0.exists():
        raise FileNotFoundError(f"Image not found: {img_path_0}")
    if not img_path_1.exists():
        raise FileNotFoundError(f"Image not found: {img_path_1}")
    
    image_0 = cv2.imread(str(img_path_0))
    image_1 = cv2.imread(str(img_path_1))
    
    if image_0 is None:
        raise ValueError(f"Could not load image: {img_path_0}")
    if image_1 is None:
        raise ValueError(f"Could not load image: {img_path_1}")
    
    image_gray_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)
    image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    
    merged_gray = ((image_gray_0.astype(np.float32) + image_gray_1.astype(np.float32)) / 2).astype(np.uint8)
    
    if len(merged_gray.shape) == 2:
        merged_color = cv2.cvtColor(merged_gray, cv2.COLOR_GRAY2BGR)
    else:
        merged_color = merged_gray.copy()
    
    contour_path = input_path / "Contour.json"
    if not contour_path.exists():
        raise FileNotFoundError(f"Contour.json not found: {contour_path}")
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    contour_arr = np.array(contour, dtype=np.int32)
    center_x = int(np.mean(contour_arr[:, 0]))
    center_y = int(np.mean(contour_arr[:, 1]))
    
    extra_data_path = input_path / "Extra Data.json"
    if not extra_data_path.exists():
        raise FileNotFoundError(f"Extra Data.json not found: {extra_data_path}")
    
    with open(extra_data_path, 'r') as f:
        extra_data = json.load(f)
        angles = extra_data.get("Polish Lines Data", {}).get("Chosen Facet PD", [])
    
    if not angles:
        raise ValueError(f"No angles found in Chosen Facet PD for {folder_name}")
    
    h, w = merged_color.shape[:2]
    max_dist = int(np.sqrt(w**2 + h**2))
    
    for angle in angles:
        angle_rad = np.deg2rad(angle + 90)
        
        dx = max_dist * np.cos(angle_rad)
        dy = max_dist * np.sin(angle_rad)
        
        pt1_x = int(center_x - dx)
        pt1_y = int(center_y - dy)
        pt2_x = int(center_x + dx)
        pt2_y = int(center_y + dy)
        
        cv2.line(merged_color, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)
    
    output_dir = Path("viz/angles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{folder_name}.png"
    cv2.imwrite(str(output_path), merged_color)
    
    print(f"Visualized: {folder_name} - {len(angles)} angles")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_lines.py <input_folder>")
        sys.exit(1)
    
    visualize_lines(sys.argv[1])

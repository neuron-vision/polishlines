import cv2
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from common_utils import erase_outer_contour, crop_from_contour

def create_patches(image, contour, num_patches=16, overlap=0.5):
    contour_arr = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour_arr)
    
    grid_size = int(np.sqrt(num_patches))
    patch_w = int(w / (grid_size * (1 - overlap) + overlap))
    patch_h = int(h / (grid_size * (1 - overlap) + overlap))
    
    step_x = int(patch_w * (1 - overlap))
    step_y = int(patch_h * (1 - overlap))
    
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch_x = x + j * step_x
            patch_y = y + i * step_y
            patch = image[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
            if patch.size > 0:
                patches.append({
                    'patch': patch,
                    'x': patch_x,
                    'y': patch_y,
                    'w': patch_w,
                    'h': patch_h,
                    'idx': i * grid_size + j
                })
    return patches

def analyze_fft_parallel_lines(patch):
    if patch.size == 0:
        return 0.0, None
    
    fft_result = np.fft.fft2(patch)
    fft_shifted = np.fft.fftshift(fft_result)
    fft_magnitude = np.abs(fft_shifted)
    
    h, w = fft_magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    angles = np.linspace(0, 180, 180)
    max_peak_value = 0.0
    best_angle = None
    
    for angle in angles:
        rad = np.deg2rad(angle)
        line_length = min(h, w) // 2
        
        y_coords = []
        x_coords = []
        for dist in range(1, line_length):
            dy = int(dist * np.sin(rad))
            dx = int(dist * np.cos(rad))
            y_coords.append(center_y + dy)
            x_coords.append(center_x + dx)
            y_coords.append(center_y - dy)
            x_coords.append(center_x - dx)
        
        valid_mask = (np.array(y_coords) >= 0) & (np.array(y_coords) < h) & \
                     (np.array(x_coords) >= 0) & (np.array(x_coords) < w)
        
        if np.sum(valid_mask) > 0:
            line_values = fft_magnitude[np.array(y_coords)[valid_mask], np.array(x_coords)[valid_mask]]
            peak_value = np.max(line_values)
            if peak_value > max_peak_value:
                max_peak_value = peak_value
                best_angle = angle
    
    return max_peak_value, best_angle

def process_detections(input_folder):
    input_path = Path(input_folder)
    folder_name = input_path.name
    
    output_dir = Path("pipeline_outputs") / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    contour_path = input_path / "Contour.json"
    if not contour_path.exists():
        raise FileNotFoundError(f"Contour.json not found: {contour_path}")
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    masked = erase_outer_contour(merged_gray, contour)
    
    patches = create_patches(masked, contour, num_patches=16, overlap=0.5)
    
    patch_results = []
    for patch_info in patches:
        patch = patch_info['patch']
        peak_value, angle = analyze_fft_parallel_lines(patch)
        
        fft_result = np.fft.fft2(patch)
        fft_shifted = np.fft.fftshift(fft_result)
        fft_magnitude = np.log(np.abs(fft_shifted) + 1)
        
        patch_results.append({
            'patch_info': patch_info,
            'peak_value': float(peak_value),
            'angle': float(angle) if angle is not None else None,
            'fft_magnitude': fft_magnitude
        })
    
    best_patch_idx = max(range(len(patch_results)), key=lambda i: patch_results[i]['peak_value'])
    best_patch = patch_results[best_patch_idx]
    
    annotated_image = merged_gray.copy()
    if len(annotated_image.shape) == 2:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
    
    best_patch_info = best_patch['patch_info']
    cv2.rectangle(annotated_image, 
                  (best_patch_info['x'], best_patch_info['y']),
                  (best_patch_info['x'] + best_patch_info['w'], 
                   best_patch_info['y'] + best_patch_info['h']),
                  (0, 255, 0), 3)
    
    cv2.imwrite(str(output_dir / "annotated_image.png"), annotated_image)
    
    for idx, result in enumerate(patch_results):
        patch_info = result['patch_info']
        patch_dir = output_dir / f"patch_{idx:02d}"
        patch_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(patch_dir / "patch.png"), patch_info['patch'])
        
        plt.figure(figsize=(8, 8))
        plt.imshow(result['fft_magnitude'], cmap='viridis')
        plt.title(f"FFT Magnitude - Peak: {result['peak_value']:.2f}, Angle: {result['angle']:.1f}°")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(patch_dir / "fft_magnitude.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        patch_json = {
            'patch_idx': idx,
            'coordinates': {
                'x': int(patch_info['x']),
                'y': int(patch_info['y']),
                'w': int(patch_info['w']),
                'h': int(patch_info['h'])
            },
            'fft_analysis': {
                'peak_value': result['peak_value'],
                'angle': result['angle'],
                'is_best_patch': idx == best_patch_idx
            }
        }
        
        with open(patch_dir / "patch_info.json", 'w') as f:
            json.dump(patch_json, f, indent=2)
    
    summary_json = {
        'folder_name': folder_name,
        'total_patches': len(patches),
        'best_patch_idx': best_patch_idx,
        'best_patch_peak_value': best_patch['peak_value'],
        'best_patch_angle': best_patch['angle'],
        'best_patch_coordinates': {
            'x': int(best_patch_info['x']),
            'y': int(best_patch_info['y']),
            'w': int(best_patch_info['w']),
            'h': int(best_patch_info['h'])
        }
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"Processed: {folder_name} - Best patch: {best_patch_idx}, Peak: {best_patch['peak_value']:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detections_pipeline.py <input_folder>")
        sys.exit(1)
    
    process_detections(sys.argv[1])

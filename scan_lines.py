import cv2
import json
import sys
import numpy as np
from pathlib import Path
from common_utils import erase_outer_contour

def create_line_kernel(angle, length=30, thickness=2):
    angle_rad = np.deg2rad(angle + 90)
    kernel_size = length * 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    center = kernel_size // 2
    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)
    
    pt1 = (int(center - dx), int(center - dy))
    pt2 = (int(center + dx), int(center + dy))
    
    cv2.line(kernel, pt1, pt2, 1.0, thickness)
    kernel = kernel - np.mean(kernel)
    kernel = kernel / (np.std(kernel) + 1e-6)
    return kernel

def scan_angle_full(image, angle, step=10, min_correlation=0.1):
    kernel = create_line_kernel(angle)
    h, w = image.shape
    kh, kw = kernel.shape
    
    if h < kh or w < kw:
        return None
    
    result_map = cv2.matchTemplate(image.astype(np.float32), kernel, cv2.TM_CCOEFF_NORMED)
    
    result_h, result_w = result_map.shape
    y_indices = np.arange(0, result_h, step)
    x_indices = np.arange(0, result_w, step)
    
    sampled_map = result_map[np.ix_(y_indices, x_indices)]
    
    mask = sampled_map > min_correlation
    y_coords, x_coords = np.where(mask)
    
    all_correlations = []
    for i in range(len(y_coords)):
        y_idx = y_coords[i]
        x_idx = x_coords[i]
        correlation = float(sampled_map[y_idx, x_idx])
        all_correlations.append({
            'angle': float(angle),
            'correlation': correlation,
            'position': (int(x_indices[x_idx]), int(y_indices[y_idx]))
        })
    
    return all_correlations, result_map

def nms_line_detections(detections, distance_threshold=10):
    if len(detections) == 0:
        return []
    
    detections = sorted(detections, key=lambda x: x['correlation'], reverse=True)
    kept = []
    
    for det in detections:
        is_too_close = False
        for kept_det in kept:
            pos_x, pos_y = det['position']
            kept_x, kept_y = kept_det['position']
            distance = np.sqrt((pos_x - kept_x)**2 + (pos_y - kept_y)**2)
            if distance < distance_threshold:
                is_too_close = True
                break
        
        if not is_too_close:
            kept.append(det)
    
    return kept

def scan_lines(input_folder, verbose=True):
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
    
    contour_path = input_path / "Contour.json"
    if not contour_path.exists():
        raise FileNotFoundError(f"Contour.json not found: {contour_path}")
    
    with open(contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    masked = erase_outer_contour(merged_gray, contour)
    
    contour_arr = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour_arr)
    cropped_image = masked[y:y+h, x:x+w]
    
    orig_h, orig_w = cropped_image.shape[:2]
    max_dim = max(orig_h, orig_w)
    
    if max_dim > 2048:
        scale = 2048.0 / max_dim
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        cropped_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if len(cropped_image.shape) == 2:
        cropped_color = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    else:
        cropped_color = cropped_image.copy()
    
    center_x_orig = int(np.mean(contour_arr[:, 0]))
    center_y_orig = int(np.mean(contour_arr[:, 1]))
    
    scale_factor = cropped_image.shape[1] / orig_w if orig_w > 0 else 1.0
    center_x = int((center_x_orig - x) * scale_factor)
    center_y = int((center_y_orig - y) * scale_factor)
    
    extra_data_path = input_path / "Extra Data.json"
    if not extra_data_path.exists():
        raise FileNotFoundError(f"Extra Data.json not found: {extra_data_path}")
    
    with open(extra_data_path, 'r') as f:
        extra_data = json.load(f)
        angles = extra_data.get("Polish Lines Data", {}).get("Chosen Facet PD", [])
        label = extra_data.get("Polish Lines Data", {}).get("User Input", "Unknown")
    
    if not angles:
        raise ValueError(f"No angles found in Chosen Facet PD for {folder_name}")
    
    all_angle_correlations = []
    angle_votes = {}
    correlation_threshold = 0.2
    
    if verbose:
        print(f"Scanning {len(angles)} angles for {folder_name}...")
    
    for idx, angle in enumerate(angles):
        if verbose:
            print(f"  Scanning angle {idx+1}/{len(angles)}: {angle:.1f}°", end='', flush=True)
        
        correlations, result_map = scan_angle_full(cropped_image, angle, step=10, min_correlation=0.1)
        if correlations is not None:
            all_angle_correlations.append({
                'angle_idx': idx,
                'angle': float(angle),
                'correlations': correlations,
                'result_map': result_map
            })
            
            strong_correlations = [c for c in correlations if c['correlation'] > correlation_threshold]
            angle_votes[float(angle)] = len(strong_correlations)
            
            if verbose:
                print(f" - Found {len(strong_correlations)} strong correlations")
        else:
            if verbose:
                print(" - Skipped (image too small)")
    
    if not angle_votes:
        raise ValueError(f"No valid correlations found for {folder_name}")
    
    selected_angle = max(angle_votes.keys(), key=lambda a: angle_votes[a])
    
    if verbose:
        print(f"Selected angle: {selected_angle:.1f}° with {angle_votes[selected_angle]} votes")
        print("Applying NMS and selecting top 5 lines...")
    
    selected_angle_data = next(d for d in all_angle_correlations if d['angle'] == selected_angle)
    all_correlations_for_angle = selected_angle_data['correlations']
    
    all_correlations_for_angle.sort(key=lambda x: x['correlation'], reverse=True)
    nms_lines = nms_line_detections(all_correlations_for_angle, distance_threshold=10)
    top_5_best_angle_lines = nms_lines[:5]
    
    if verbose:
        print(f"Selected {len(top_5_best_angle_lines)} lines after NMS")
    
    h, w = cropped_color.shape[:2]
    max_dist = int(np.sqrt(w**2 + h**2))
    
    best_angle_rad = np.deg2rad(selected_angle + 90)
    
    for rank, line_result in enumerate(top_5_best_angle_lines):
        pos_x, pos_y = line_result['position']
        dx = max_dist * np.cos(best_angle_rad)
        dy = max_dist * np.sin(best_angle_rad)
        
        pt1_x = int(pos_x - dx)
        pt1_y = int(pos_y - dy)
        pt2_x = int(pos_x + dx)
        pt2_y = int(pos_y + dy)
        
        cv2.line(cropped_color, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 3)
    
    text_y = 40
    cv2.putText(cropped_color, f"Selected Angle: {selected_angle:.1f}° (Votes: {angle_votes[selected_angle]}) - Top 5 Lines:", 
               (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    text_y += 40
    
    for rank, line_result in enumerate(top_5_best_angle_lines):
        pos_x, pos_y = line_result['position']
        text = f"  #{rank+1}: @ ({pos_x},{pos_y}) - Corr: {line_result['correlation']:.3f}"
        cv2.putText(cropped_color, text, (20, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_y += 30
    
    output_dir = Path("viz/correlations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_image_path = output_dir / f"{folder_name}_{label}.png"
    cv2.imwrite(str(output_image_path), cropped_color)
    
    print(f"Scanned: {folder_name} ({label}) - Selected: {selected_angle:.1f}° (Votes: {angle_votes[selected_angle]})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_lines.py <input_folder> [--verbose]")
        sys.exit(1)
    
    verbose = '--verbose' in sys.argv
    scan_lines(sys.argv[1], verbose=verbose)

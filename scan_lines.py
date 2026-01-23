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

def point_to_contour_distance(point, contour, image_shape):
    pos_x, pos_y = point
    h, w = image_shape
    
    if pos_x < 0 or pos_x >= w or pos_y < 0 or pos_y >= h:
        return float('inf')
    
    contour_arr = np.array(contour, dtype=np.int32)
    distances = cv2.pointPolygonTest(contour_arr, (pos_x, pos_y), True)
    
    return abs(distances)

def nms_line_detections(detections, distance_threshold=10, contour=None, contour_distance_threshold=20, image_shape=None):
    if len(detections) == 0:
        return []
    
    detections = sorted(detections, key=lambda x: x['correlation'], reverse=True)
    kept = []
    
    for det in detections:
        pos_x, pos_y = det['position']
        
        if contour is not None and image_shape is not None:
            dist_to_contour = point_to_contour_distance((pos_x, pos_y), contour, image_shape)
            if dist_to_contour < contour_distance_threshold:
                continue
        
        is_too_close = False
        for kept_det in kept:
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
    
    preprocessed_dir = Path("preprocessed_scan_lines")
    preprocessed_image_path = preprocessed_dir / f"{folder_name}.png"
    preprocessed_contour_path = preprocessed_dir / f"{folder_name}_contour.json"
    preprocessed_extra_path = preprocessed_dir / f"{folder_name}_extra.json"
    
    if not preprocessed_image_path.exists():
        raise FileNotFoundError(f"Preprocessed image not found: {preprocessed_image_path}")
    if not preprocessed_contour_path.exists():
        raise FileNotFoundError(f"Preprocessed contour not found: {preprocessed_contour_path}")
    
    cropped_image = cv2.imread(str(preprocessed_image_path), cv2.IMREAD_GRAYSCALE)
    if cropped_image is None:
        raise ValueError(f"Could not load preprocessed image: {preprocessed_image_path}")
    
    with open(preprocessed_contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    if len(cropped_image.shape) == 2:
        cropped_color = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    else:
        cropped_color = cropped_image.copy()
    
    contour_arr = np.array(contour, dtype=np.int32)
    center_x = int(np.mean(contour_arr[:, 0]))
    center_y = int(np.mean(contour_arr[:, 1]))
    
    if preprocessed_extra_path.exists():
        with open(preprocessed_extra_path, 'r') as f:
            extra_data = json.load(f)
    else:
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
    nms_lines = nms_line_detections(
        all_correlations_for_angle, 
        distance_threshold=10,
        contour=contour,
        contour_distance_threshold=20,
        image_shape=cropped_image.shape
    )
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

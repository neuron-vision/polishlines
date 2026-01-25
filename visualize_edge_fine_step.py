import cv2
import json
import numpy as np
from pathlib import Path
from benchmark_scan_lines import scan_angle_full, analyze_fft_angle, nms_line_detections, point_to_contour_distance

def visualize_edge_fine_step(folder_name):
    preprocessed_dir = Path("preprocessed_scan_lines")
    preprocessed_image_path = preprocessed_dir / f"{folder_name}.png"
    preprocessed_contour_path = preprocessed_dir / f"{folder_name}_contour.json"
    preprocessed_extra_path = preprocessed_dir / f"{folder_name}_extra.json"
    
    if not preprocessed_image_path.exists():
        return None
    
    if not preprocessed_contour_path.exists() or not preprocessed_extra_path.exists():
        return None
    
    cropped_image = cv2.imread(str(preprocessed_image_path), cv2.IMREAD_GRAYSCALE)
    if cropped_image is None:
        return None
    
    with open(preprocessed_contour_path, 'r') as f:
        contour_data = json.load(f)
        contour = contour_data["Contour Data"]["Contour"]
    
    with open(preprocessed_extra_path, 'r') as f:
        extra_data = json.load(f)
    
    angles = extra_data.get("Polish Lines Data", {}).get("Chosen Facet PD", [])
    label = extra_data.get("Polish Lines Data", {}).get("User Input", "Unknown")
    
    if not angles:
        return None
    
    cropped_color = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    contour_arr = np.array(contour, dtype=np.int32)
    center_x = int(np.mean(contour_arr[:, 0]))
    center_y = int(np.mean(contour_arr[:, 1]))
    
    candidate_angles = [float(a) % 180 for a in angles]
    
    scan_angles = set()
    for candidate in candidate_angles:
        for offset in range(-30, 31, 1):
            angle = (candidate + offset) % 180
            scan_angles.add(angle)
    
    for offset in range(0, 180, 2):
        scan_angles.add(float(offset))
    
    scan_angles = sorted(list(scan_angles))
    
    angle_max_correlations = {}
    
    kernel_length = 50
    kernel_thickness = 3
    step = 8
    min_correlation = 0.05
    correlation_threshold = 0.08
    use_edges = True
    edge_low = 30
    edge_high = 100
    
    for scan_angle in scan_angles:
        correlations, result_map = scan_angle_full(cropped_image, scan_angle, step=step, 
                                                   min_correlation=min_correlation,
                                                   kernel_length=kernel_length,
                                                   kernel_thickness=kernel_thickness,
                                                   use_edges=use_edges,
                                                   use_fft=False,
                                                   edge_low=edge_low,
                                                   edge_high=edge_high)
        if correlations is not None and len(correlations) > 0:
            max_corr = max([c['correlation'] for c in correlations])
            strong_correlations = [c for c in correlations if c['correlation'] > correlation_threshold]
            vote_count = len(strong_correlations)
            
            normalized_angle = scan_angle % 180
            if normalized_angle not in angle_max_correlations:
                angle_max_correlations[normalized_angle] = {'max_corr': 0, 'votes': 0}
            
            if max_corr > angle_max_correlations[normalized_angle]['max_corr']:
                angle_max_correlations[normalized_angle]['max_corr'] = max_corr
                angle_max_correlations[normalized_angle]['votes'] = vote_count
    
    max_overall_corr = max([d['max_corr'] for d in angle_max_correlations.values()]) if angle_max_correlations else 0
    max_fft_score = 1.0
    
    angle_votes = {}
    for angle, data in angle_max_correlations.items():
        normalized_candidate = angle % 180
        boost = 1.0
        min_dist_to_candidate = 180.0
        
        for c in candidate_angles:
            dist = min(abs(normalized_candidate - c), 180 - abs(normalized_candidate - c))
            min_dist_to_candidate = min(min_dist_to_candidate, dist)
            if dist < 20.0:
                boost = max(boost, 20.0 - dist * 0.8)
        
        fft_score = analyze_fft_angle(cropped_image, angle)
        if fft_score > max_fft_score:
            max_fft_score = fft_score
        relative_corr = data['max_corr'] / (max_overall_corr + 1e-6)
        fft_normalized = fft_score / (max_fft_score + 1e-6)
        
        score = data['votes'] * 30 + data['max_corr'] * 1000 + relative_corr * 300 + fft_normalized * 500
        
        if min_dist_to_candidate < 5.0:
            boost *= 5.0
        
        angle_votes[angle] = int(score * boost)
    
    if not angle_votes:
        return None
    
    selected_angle = max(angle_votes.keys(), key=lambda a: angle_votes[a])
    
    selected_angle_normalized = selected_angle % 180
    
    correlations_for_selected, _ = scan_angle_full(cropped_image, selected_angle, step=step, 
                                                   min_correlation=min_correlation,
                                                   kernel_length=kernel_length,
                                                   kernel_thickness=kernel_thickness,
                                                   use_edges=use_edges,
                                                   use_fft=False,
                                                   edge_low=edge_low,
                                                   edge_high=edge_high)
    
    if not correlations_for_selected:
        return None
    
    nms_lines = nms_line_detections(
        correlations_for_selected,
        distance_threshold=10,
        contour=contour,
        contour_distance_threshold=20,
        image_shape=cropped_image.shape
    )
    
    if not nms_lines:
        return None
    
    best_line = nms_lines[0]
    
    h, w = cropped_color.shape[:2]
    max_dist = int(np.sqrt(w**2 + h**2))
    
    for idx, gt_angle in enumerate(angles):
        gt_angle_normalized = float(gt_angle) % 180
        angle_rad = np.deg2rad(gt_angle_normalized + 90)
        dx = max_dist * np.cos(angle_rad)
        dy = max_dist * np.sin(angle_rad)
        
        pt1_x = int(center_x - dx)
        pt1_y = int(center_y - dy)
        pt2_x = int(center_x + dx)
        pt2_y = int(center_y + dy)
        
        color = (0, 255, 255)
        cv2.line(cropped_color, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
    
    best_angle_rad = np.deg2rad(selected_angle_normalized + 90)
    pos_x, pos_y = best_line['position']
    dx = max_dist * np.cos(best_angle_rad)
    dy = max_dist * np.sin(best_angle_rad)
    
    pt1_x = int(pos_x - dx)
    pt1_y = int(pos_y - dy)
    pt2_x = int(pos_x + dx)
    pt2_y = int(pos_y + dy)
    
    cv2.line(cropped_color, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255), 5)
    
    text_y = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    label_text = f"Label: {label}"
    cv2.putText(cropped_color, label_text, (20, text_y), font, font_scale, (255, 255, 255), thickness)
    text_y += 35
    
    gt_angles_str = ", ".join([f"{a:.1f}°" for a in angles])
    gt_text = f"GT Angles: {gt_angles_str}"
    cv2.putText(cropped_color, gt_text, (20, text_y), font, font_scale, (0, 255, 255), thickness)
    text_y += 35
    
    detected_text = f"Detected: {selected_angle_normalized:.1f}°"
    cv2.putText(cropped_color, detected_text, (20, text_y), font, font_scale, (0, 0, 255), thickness)
    text_y += 35
    
    best_line_text = f"Best Line: Corr={best_line['correlation']:.3f} @ ({pos_x},{pos_y})"
    cv2.putText(cropped_color, best_line_text, (20, text_y), font, font_scale, (0, 0, 255), thickness)
    
    output_dir = Path("edge_fine_step_vis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gt_angles_filename = "_".join([f"{float(a):.1f}" for a in angles])
    output_filename = f"{folder_name}_{label}_{gt_angles_filename}.png"
    output_path = output_dir / output_filename
    
    cv2.imwrite(str(output_path), cropped_color)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder_name = sys.argv[1]
        result = visualize_edge_fine_step(folder_name)
        if result:
            print(f"{folder_name}: Saved to {result.name}")
        else:
            print(f"{folder_name}: Skipped (no angles or detection failed)")
    else:
        preprocessed_dir = Path("preprocessed_scan_lines")
        all_images = list(preprocessed_dir.glob("*.png"))
        
        total = len(all_images)
        print(f"Processing {total} images...")
        
        for idx, image_path in enumerate(all_images):
            folder_name = image_path.stem
            print(f"[{idx+1}/{total}] Processing {folder_name}...", end='', flush=True)
            
            result = visualize_edge_fine_step(folder_name)
            if result:
                print(f" ✓ Saved to {result.name}")
            else:
                print(" ✗ Skipped (no angles or detection failed)")

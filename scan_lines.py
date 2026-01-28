import cv2
import os
import json
import sys
import numpy as np
from pathlib import Path
from common_utils import erase_outer_contour

def create_line_kernel(angle, length=100, thickness=2):
    angle_rad = np.deg2rad(angle)
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

if __name__ == "__main__":
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed_data/Has_PL/1529")
    data = json.load(open(input_path / "data.json"))
    image = cv2.imread(data['image_path'], cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    w, h= image.shape
    kernel_length = (h**2 + w**2)**0.5
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask = cv2.imread(data['mask_path'], cv2.IMREAD_GRAYSCALE)
    
    thickness = 2
    for angle in data['angles']:
        angle_rad = np.deg2rad(angle + 90)
        kernel_size = kernel_length * 2
        
        center = kernel_size // 2
        dx = kernel_length * np.cos(angle_rad)
        dy = kernel_length * np.sin(angle_rad)
        
        pt1 = (int(center - dx), int(center - dy))
        pt2 = (int(center + dx), int(center + dy))
        
        cv2.line(color_image, pt1, pt2, (0, 0, 255), thickness)

    cv2.imwrite(str(input_path / "image_with_lines.png"), color_image)
    cmd = f"open {input_path / 'image_with_lines.png'}"
        
        


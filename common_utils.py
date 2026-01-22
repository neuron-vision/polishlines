import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_IMAGE_SIZE = (512, 512)

def crop_from_contour(image, contour):
    contour = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y+h, x:x+w]
    return cropped

def erase_outer_contour(image, contour):
    contour = np.array(contour, dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked

if __name__ == "__main__":
    sample_dir = Path("data/train/0")
    
    img_path = sample_dir / "0.png"
    contour_path = sample_dir / "Contour.json"
    
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    with open(contour_path, 'r') as f:
        data = json.load(f)
        contour = data["Contour Data"]["Contour"]
    
    masked = erase_outer_contour(image_gray, contour)
    cropped = crop_from_contour(masked, contour)
    contour_arr = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour_arr)
    print(f"Original image shape: {image_gray.shape}")
    print(f"Erased image shape: {masked.shape}")
    print(f"Cropped image shape: {cropped.shape}")
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    resized = cv2.resize(cropped, DEFAULT_IMAGE_SIZE)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(image_gray, cmap='gray')
    axes[0, 0].set_title("Original Grayscale Image")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(masked, cmap='gray')
    axes[0, 1].set_title("Erased Image")
    axes[0, 1].axis('off')
    axes[1, 0].imshow(cropped, cmap='gray')
    axes[1, 0].set_title("Cropped Image")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(resized, cmap='gray')
    axes[1, 1].set_title("Resized Image")
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()
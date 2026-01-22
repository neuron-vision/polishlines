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

if __name__ == "__main__":
    sample_dir = Path("data/train/0")
    
    img_path = sample_dir / "0.png"
    contour_path = sample_dir / "Contour.json"
    
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    with open(contour_path, 'r') as f:
        data = json.load(f)
        contour = data["Contour Data"]["Contour"]
    
    cropped = crop_from_contour(image, contour)
    contour_arr = np.array(contour, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour_arr)
    print(f"Original image shape: {image.shape}")
    print(f"Cropped image shape: {cropped.shape}")
    print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    square_cropped = cv2.resize(cropped, DEFAULT_IMAGE_SIZE)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(cropped_rgb)
    axes[1].set_title("Cropped Image")
    axes[1].axis('off')
    axes[2].imshow(square_cropped)
    axes[2].set_title("Square Cropped Image")
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()
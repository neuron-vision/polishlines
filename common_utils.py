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
    train_dir = Path("data/train")
    
    has_pl_samples = []
    no_pl_samples = []
    
    for subdir in sorted(train_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        extra_data_path = subdir / "Extra Data.json"
        if not extra_data_path.exists():
            continue
        
        with open(extra_data_path, 'r') as f:
            extra_data = json.load(f)
            label = extra_data.get("Polish Lines Data", {}).get("User Input", "")
        
        if label == "Has_PL" and len(has_pl_samples) < 3:
            has_pl_samples.append(subdir)
        elif label == "No_PL" and len(no_pl_samples) < 3:
            no_pl_samples.append(subdir)
        
        if len(has_pl_samples) >= 3 and len(no_pl_samples) >= 3:
            break
    
    all_samples = has_pl_samples + no_pl_samples
    all_labels = ["Has_PL"] * len(has_pl_samples) + ["No_PL"] * len(no_pl_samples)
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    fig.suptitle("3 with Polish Lines (Has_PL) | 3 without Polish Lines (No_PL) - Images and FFT", fontsize=16, fontweight='bold')
    
    processed_images = []
    
    for idx, (sample_dir, label) in enumerate(zip(all_samples, all_labels)):
        img_path = sample_dir / "0.png"
        contour_path = sample_dir / "Contour.json"
        
        if not img_path.exists() or not contour_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        with open(contour_path, 'r') as f:
            data = json.load(f)
            contour = data["Contour Data"]["Contour"]
        
        masked = erase_outer_contour(image_gray, contour)
        cropped = crop_from_contour(masked, contour)
        resized = cv2.resize(cropped, DEFAULT_IMAGE_SIZE)
        
        fft_resized = np.fft.fft2(resized)
        fft_shifted = np.fft.fftshift(fft_resized)
        fft_magnitude = np.log(np.abs(fft_shifted) + 1)
        
        processed_images.append((resized, fft_magnitude, sample_dir.name, label))
    
    for idx, (resized, fft_magnitude, folder_name, label) in enumerate(processed_images):
        row = 0 if label == "Has_PL" else 1
        col = idx if label == "Has_PL" else idx - 3
        
        axes[row, col * 2].imshow(resized, cmap='gray')
        axes[row, col * 2].set_title(f"{folder_name} - {label}", fontsize=10, fontweight='bold')
        axes[row, col * 2].axis('off')
        
        axes[row, col * 2 + 1].imshow(fft_magnitude, cmap='viridis')
        axes[row, col * 2 + 1].set_title(f"FFT - {label}", fontsize=10)
        axes[row, col * 2 + 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle("FFT Comparison: Top 3 No_PL | Bottom 3 Has_PL", fontsize=16, fontweight='bold')
    
    no_pl_ffts = []
    has_pl_ffts = []
    
    for idx, (sample_dir, label) in enumerate(zip(all_samples, all_labels)):
        img_path = sample_dir / "0.png"
        contour_path = sample_dir / "Contour.json"
        
        if not img_path.exists() or not contour_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        with open(contour_path, 'r') as f:
            data = json.load(f)
            contour = data["Contour Data"]["Contour"]
        
        masked = erase_outer_contour(image_gray, contour)
        cropped = crop_from_contour(masked, contour)
        resized = cv2.resize(cropped, DEFAULT_IMAGE_SIZE)
        
        fft_resized = np.fft.fft2(resized)
        fft_shifted = np.fft.fftshift(fft_resized)
        fft_magnitude = np.log(np.abs(fft_shifted) + 1)
        
        if label == "No_PL":
            no_pl_ffts.append(fft_magnitude)
        else:
            has_pl_ffts.append(fft_magnitude)
    
    for idx in range(3):
        if idx < len(no_pl_ffts):
            axes2[0, idx].imshow(no_pl_ffts[idx], cmap='viridis')
            axes2[0, idx].axis('off')
        
        if idx < len(has_pl_ffts):
            axes2[1, idx].imshow(has_pl_ffts[idx], cmap='viridis')
            axes2[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
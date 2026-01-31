import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import json
import os
import sys

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_utils import CNN_PREPROCESSED_FOLDER

class HorizontalAEDataset(Dataset):
    def __init__(self, root_dir=CNN_PREPROCESSED_FOLDER):
        self.samples = []
        for data_json in Path(root_dir).glob("*/*/data.json"):
            with open(data_json) as f:
                data = json.load(f)
            for a in data['angles']:
                if 'image' in a and 'highpassed_image' in a and 'mask' in a:
                    img_path = Path(a['image'])
                    hp_path = Path(a['highpassed_image'])
                    mask_path = Path(a['mask'])
                    if img_path.exists() and hp_path.exists() and mask_path.exists():
                        self.samples.append((img_path, hp_path, mask_path))
        print(f"Dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, hp_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert('L')
        hp = Image.open(hp_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        # Ensure 1024x1024
        if img.size != (1024, 1024):
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            hp = hp.resize((1024, 1024), Image.Resampling.LANCZOS)
            mask = mask.resize((1024, 1024), Image.Resampling.NEAREST)
            
        img_np = np.array(img).astype(np.float32) / 255.0
        hp_np = np.array(hp).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0
        
        # Stack into 3 channels: [Image, Highpass, Mask]
        data = np.stack([img_np, hp_np, mask_np], axis=0)
        return torch.from_numpy(data)

def get_dataloader(batch_size=4, shuffle=True):
    dataset = HorizontalAEDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    loader = get_dataloader(batch_size=2)
    for i, batch in enumerate(loader):
        print(f"Batch {i} shape: {batch.shape}")
        if i >= 2: break

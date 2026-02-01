#!/Users/ishay/projects/magic/policy/functions/venv/bin/python
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import json
import os
import sys
import shutil
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.export_embedder import HorizontalEmbedder
from common_utils import CNN_PREPROCESSED_FOLDER

POSITIVE_LABELS = {"Has_PL"}
EMBEDDING_CACHE_DIR = Path("embeddings/auto_encoder")

class PolishLinesDataset:
    def __init__(self, root_dir=CNN_PREPROCESSED_FOLDER):
        self.folders = []
        for label_dir in Path(root_dir).glob("*"):
            if not label_dir.is_dir(): continue
            label_str = label_dir.name
            binary_label = 1 if label_str in POSITIVE_LABELS else 0
            folders_list = list(label_dir.glob("*"))
            for folder_dir in tqdm(folders_list, desc="Processing folders"):
                data_json = folder_dir / "data.json"
                if data_json.exists():
                    with open(data_json) as f:
                        data = json.load(f)
                    img_paths = []
                    for a in data['angles']:
                        if 'image' in a and 'highpassed_image' in a and 'mask' in a:
                            p = Path(a['image'])
                            hp = Path(a['highpassed_image'])
                            m = Path(a['mask'])
                            if p.exists() and hp.exists() and m.exists():
                                img_paths.append((p, hp, m))
                    if img_paths:
                        self.folders.append({
                            'name': folder_dir.name,
                            'images': img_paths,
                            'label': binary_label
                        })
        print(f"Initialized with {len(self.folders)} folders.")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        return self.folders[idx]

    def get_folder_embeddings(self, folder, embedder, device):
        embeddings = []
        for img_path, hp_path, mask_path in folder['images']:
            hp = Image.open(hp_path).convert('L')
            if hp.size != (1024, 1024):
                hp = hp.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            hp_np = np.array(hp).astype(np.float32) / 255.0
            data = hp_np[np.newaxis, ...]
            tensor = torch.from_numpy(data).unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = embedder(tensor).cpu().numpy()
            embeddings.append(emb)
        
        # Aggregate: Max across all rotated images for this folder
        return np.max(np.vstack(embeddings), axis=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if 'force' in sys.argv:
        shutil.rmtree(EMBEDDING_CACHE_DIR, ignore_errors=True)
    print(f"Using device: {device}")
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    embedder = HorizontalEmbedder().to(device)
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False

    ds = PolishLinesDataset()
    X, y = [], []
    cache_metadata = []
    already_cached = 0
    
    print("Extracting features and caching...")
    try:
        progress_bar = tqdm(ds.folders, desc="Extracting features")
        for folder in progress_bar:
            emb_path = EMBEDDING_CACHE_DIR / f"{folder['name']}.npy"
            if emb_path.exists():
                already_cached += 1
                continue
            features = ds.get_folder_embeddings(folder, embedder, device)
            # Cache the embedding
            np.save(emb_path, features)
            progress_bar.set_postfix(already_cached=already_cached)
    except KeyboardInterrupt:
        print("Extraction interrupted by user")


    print(f"Done extracting features. {already_cached} folders already cached.")
    for folder in tqdm(ds.folders, desc="Loading cached embeddings"):
        emb_path = EMBEDDING_CACHE_DIR / f"{folder['name']}.npy"
        if emb_path.exists():
            features = np.load(emb_path)
            len_features = len(features)
            X.append(features)
            y.append(folder['label'])
    
    # Save metadata

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Training XGBoost...")
    clf = XGBClassifier(n_estimators=4000, max_depth=25, learning_rate=0.05, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save the classifier
    with open("polish_lines_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Classifier saved to polish_lines_classifier.pkl")

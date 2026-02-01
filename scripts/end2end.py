import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.export_embedder import HorizontalEmbedder
from scripts.data_loader import HorizontalAEDataset

POSITIVE_LABELS = {"Has_PL"}

class PolishLinesNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # HorizontalEmbedder outputs 1024 features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = HorizontalAEDataset(return_labels=True)
    if len(full_dataset) == 0:
        print("Dataset is empty.")
        return

    # Map labels to binary
    indices = np.arange(len(full_dataset))
    labels = []
    for i in tqdm(range(len(full_dataset)), desc="Mapping labels"):
        _, label_str = full_dataset[i]
        labels.append(1 if label_str in POSITIVE_LABELS else 0)
    labels = np.array(labels)

    # Split dataset
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Initialize model
    encoder = HorizontalEmbedder()
    # Try to load AE weights if they exist
    ae_weights = "horizontal_embedder_ae.pt"
    if os.path.exists(ae_weights):
        print(f"Loading pretrained encoder weights from {ae_weights}")
        encoder.load_state_dict(torch.load(ae_weights, map_location='cpu', weights_only=True))
    
    model = PolishLinesNet(encoder).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    print("Starting End-to-End training...")
    
    try:
        for epoch in range(100):
            model.train()
            train_loss = 0
            for batch, label_strs in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                batch = batch.to(device)
                targets = torch.tensor([1.0 if l in POSITIVE_LABELS else 0.0 for l in label_strs], device=device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch, label_strs in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                    batch = batch.to(device)
                    targets = [1 if l in POSITIVE_LABELS else 0 for l in label_strs]
                    outputs = torch.sigmoid(model(batch)).cpu().numpy()
                    val_preds.extend((outputs > 0.5).astype(int).flatten())
                    val_targets.extend(targets)
            
            acc = accuracy_score(val_targets, val_preds)
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
            
            scheduler.step(acc)
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "best_polish_lines_net.pt")
                print(f"New best model saved with accuracy: {best_acc:.4f}")
            
            if best_acc >= 0.90:
                print("Target accuracy reached!")
                break
                
    except KeyboardInterrupt:
        print("Training interrupted.")

    print(f"Final Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()

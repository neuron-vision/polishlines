import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.export_embedder import HorizontalEmbedder
from scripts.data_loader import get_dataloader

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)
        layers = []
        in_c = 1024
        channels = [512, 512, 256, 256, 128, 64, 32, 16, 8, 1]
        for i, out_c in enumerate(channels):
            layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=(4, 11), stride=(2, 1), padding=(1, 5)))
            layers.append(nn.BatchNorm2d(out_c))
            if i < len(channels) - 1:
                layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.decoder_h = nn.Sequential(*layers)
        self.upsample_w = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.fc(x).view(-1, 1024, 1, 1)
        x = self.decoder_h(x)
        x = self.upsample_w(x)
        return x

class AutoEmbedder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

def evaluate_accuracy(encoder, device):
    encoder.eval()
    loader = get_dataloader(batch_size=1, shuffle=False, return_labels=True)
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch, label in loader:
            batch = batch.to(device)
            emb = encoder(batch).cpu().numpy()
            embeddings.append(emb)
            labels.extend(label)
    
    if not embeddings:
        return 0.0
    
    X = np.vstack(embeddings)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    if len(np.unique(y)) < 2:
        return 0.0
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    encoder.train()
    return acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_dataloader(batch_size=8, shuffle=True)
    encoder = HorizontalEmbedder().to(device)
    model = AutoEmbedder(encoder).to(device)
    
    if os.path.exists("auto_embedder.pt"):
        print("Loading full auto-encoder state...")
        try:
            model.load_state_dict(torch.load("auto_embedder.pt", map_location=device))
        except:
            print("Failed to load auto_embedder.pt, starting fresh.")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training loop...")
    accuracy = 0.0
    epoch = 0
    try:
        while accuracy < 0.90:
            model.train()
            total_loss = 0
            progress_bar = tqdm(loader, desc=f"Epoch {epoch} Training")
            for i, batch in enumerate(progress_bar):
                inputs = batch.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                if device.type == 'mps' and i % 10 == 0:
                    torch.mps.empty_cache()
            
            print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(loader):.6f}")
            accuracy = evaluate_accuracy(encoder, device)
            print(f"Current Accuracy: {accuracy:.4f}")
            
            torch.save(model.state_dict(), "auto_embedder.pt")
            torch.save(encoder.state_dict(), "horizontal_embedder_ae.pt")
            epoch += 1
            if epoch > 100:
                print("Max epochs reached.")
                break
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "auto_embedder.pt")
        torch.save(encoder.state_dict(), "horizontal_embedder_ae.pt")
        print("Training interrupted by user")

    print(f"Final Accuracy: {accuracy:.4f}. Model saved.")

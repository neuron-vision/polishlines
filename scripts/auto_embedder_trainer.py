import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm
import shutil
from pathlib import Path

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
        channels = [512, 512, 256, 256, 128, 64, 32, 16, 8, 3]
        for i, out_c in enumerate(channels):
            # Mirroring HorizontalEmbedder with smaller kernel
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_dataloader(batch_size=1, shuffle=True)
    encoder = HorizontalEmbedder().to(device)
    model = AutoEmbedder(encoder).to(device)
    
    if os.path.exists("auto_embedder.pt"):
        print("Loading full auto-encoder state...")
        model.load_state_dict(torch.load("auto_embedder.pt", map_location=device))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    #lways allow keyboard interrupt
    try:
        for epoch in tqdm(range(4), desc="Training"):
            total_loss = 0
            progress_bar = tqdm(loader, desc="Training")
            for i, batch in enumerate(progress_bar):
                inputs = batch.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Apply mask (channel 2) to both output and input for loss calculation
                mask = inputs[:, 2:3, :, :]
                loss = criterion(outputs * mask, inputs * mask)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                progress_bar.set_postfix(loss=loss.item(), epoch=epoch)
                if device.type == 'mps' and i % 10 == 0:
                    torch.mps.empty_cache()
            print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(loader):.6f}")
    except KeyboardInterrupt:
        print("Training interrupted by user")


    torch.save(model.state_dict(), "auto_embedder.pt")
    torch.save(encoder.state_dict(), "horizontal_embedder_ae.pt")
    
    EMBEDDING_CACHE_DIR = Path("embeddings/auto_encoder")
    shutil.rmtree(EMBEDDING_CACHE_DIR, ignore_errors=True)
    print("Training complete. Model saved to horizontal_embedder_ae.pt and auto_embedder.pt")

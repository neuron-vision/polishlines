import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from common_utils import CNN_PREPROCESSED_FOLDER, load_meta_params

POSITIVE_LABELS = {"Has_PL"}
meta_params = load_meta_params()

class AngleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (angle_image_path, binary_label)
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class SampleDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (folder_name, [angle_paths], binary_label)
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        folder_name, angle_paths, label = self.samples[idx]
        images = []
        for p in angle_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        return torch.stack(images), label, folder_name

def load_all_samples():
    samples = []
    for data_json in CNN_PREPROCESSED_FOLDER.glob("*/*/data.json"):
        with open(data_json) as f:
            data = json.load(f)
        label_str = data['label']
        binary_label = 1 if label_str in POSITIVE_LABELS else 0
        folder_name = data_json.parent.name
        angle_paths = [Path(a['image_path']) for a in data['angles']]
        samples.append((folder_name, angle_paths, binary_label))
    return samples

def create_angle_samples(samples):
    angle_samples = []
    for folder_name, angle_paths, label in samples:
        for p in angle_paths:
            angle_samples.append((p, label))
    return angle_samples

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

class CNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

def eval_sample_level(model, samples, transform, device, verbose=False, agg='max'):
    model.eval()
    correct, total = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    misses = []
    with torch.no_grad():
        for folder_name, angle_paths, label in samples:
            probs = []
            for p in angle_paths:
                img = Image.open(p).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                out = torch.sigmoid(model(img)).item()
                probs.append(out)
            if agg == 'max':
                score = max(probs)
            elif agg == 'mean':
                score = np.mean(probs)
            else:
                score = np.median(probs)
            pred = 1 if score > 0.5 else 0
            if pred == label:
                correct += 1
            else:
                misses.append((folder_name, pred, label, score))
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1
            total += 1
    if verbose and misses:
        print("Misclassified:")
        for folder, pred, true, prob in misses:
            print(f"  {folder}: pred={pred} true={true} prob={prob:.3f}")
    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return dict(acc=acc, precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Params: res={meta_params['cnn_min_resolution']} bs={meta_params['cnn_batch_size']} lr={meta_params['cnn_lr']} drop={meta_params['cnn_dropout']}")
    
    all_samples = load_all_samples()
    print(f"Total samples: {len(all_samples)}")
    labels = [s[2] for s in all_samples]
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
    
    train_samples, val_samples = train_test_split(all_samples, test_size=0.2, stratify=labels, random_state=42)
    train_angle_samples = create_angle_samples(train_samples)
    val_angle_samples = create_angle_samples(val_samples)
    print(f"Train angles: {len(train_angle_samples)}, Val angles: {len(val_angle_samples)}")
    
    res = meta_params['cnn_min_resolution']
    train_transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = AngleDataset(train_angle_samples, train_transform)
    val_ds = AngleDataset(val_angle_samples, val_transform)
    train_loader = DataLoader(train_ds, batch_size=meta_params['cnn_batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=meta_params['cnn_batch_size'], shuffle=False, num_workers=0)
    
    model = CNN(dropout=meta_params['cnn_dropout']).to(device)
    criterion = FocalLoss(alpha=meta_params['cnn_focal_alpha'], gamma=meta_params['cnn_focal_gamma'])
    optimizer = optim.AdamW(model.parameters(), lr=meta_params['cnn_lr'], weight_decay=meta_params['cnn_weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=meta_params['cnn_lr']*10, epochs=meta_params['cnn_epochs'], steps_per_epoch=len(train_loader))
    
    best_acc = 0
    patience_counter = 0
    for epoch in range(meta_params['cnn_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        for _ in range(len(train_loader)):
            scheduler.step()
        
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        val_metrics = eval_sample_level(model, val_samples, val_transform, device)
        print(f"E{epoch+1:02d} TL:{train_loss:.4f} TA:{train_acc:.3f} VL:{val_loss:.4f} VA:{val_acc:.3f} "
              f"Acc:{val_metrics['acc']:.3f} P:{val_metrics['precision']:.3f} R:{val_metrics['recall']:.3f} F1:{val_metrics['f1']:.3f}")
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save(model.state_dict(), "best_cnn.pt")
            print(f"  Saved (Acc={best_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= meta_params['cnn_patience']:
                print("Early stopping")
                break
    
    model.load_state_dict(torch.load("best_cnn.pt"))
    for agg in ['max', 'mean', 'median']:
        metrics = eval_sample_level(model, val_samples, val_transform, device, verbose=(agg=='max'), agg=agg)
        print(f"\n{agg.upper()}: Acc:{metrics['acc']:.3f} P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1']:.3f}")
        print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

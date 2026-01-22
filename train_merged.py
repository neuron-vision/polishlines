import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import subprocess
import os

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def load_data(data_dir):
    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    
    for json_file in sorted(data_dir.glob("*.json")):
        img_file = json_file.with_suffix('.png')
        if not img_file.exists():
            continue
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            label = data.get("Polish Lines Data_User Input")
            if label:
                image_paths.append(str(img_file))
                labels.append(label)
    
    return image_paths, labels

def get_model(arch_name, num_classes):
    if arch_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    return model

def notify_macos(message, accuracy=None):
    if accuracy is not None:
        full_message = f"Done training and got {accuracy:.2f}% accuracy"
    else:
        full_message = message
    
    subprocess.run(['osascript', '-e', f'display notification "{full_message}" with title "Training Complete"'])
    subprocess.run(['say', full_message])

def finetune_merged(data_dir="processed/train", epochs=1, batch_size=32, lr=0.001, arch_name='mobilenet_v3_small'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    image_paths, labels = load_data(data_dir)
    print(f"Loaded {len(image_paths)} samples")
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {label_encoder.classes_}")
    print(f"Number of classes: {num_classes}")
    
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 samples for train/val split, got {len(image_paths)}")
    
    if len(image_paths) < 5:
        train_paths, val_paths, train_labels, val_labels = (
            image_paths, [], encoded_labels, []
        )
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageDataset(train_paths, train_labels, transform=transform_train)
    val_dataset = ImageDataset(val_paths, val_labels, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = get_model(arch_name, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_model_state = None
    training_history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss/len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc
        }
        training_history.append(epoch_data)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'num_classes': num_classes,
                'arch_name': arch_name
            }, f'{arch_name}_merged_finetuned.pth')
            print(f"Saved best model with val acc: {val_acc:.2f}%")
        print()
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    if len(val_paths) > 0:
        model.load_state_dict(best_model_state)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        class_names = label_encoder.classes_
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)
        
        benchmark_results = {
            'overall_accuracy': float(accuracy),
            'best_val_accuracy': float(best_val_acc),
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            },
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'training_history': training_history,
            'model_info': {
                'model': arch_name,
                'num_classes': num_classes,
                'classes': class_names.tolist(),
                'train_samples': len(train_paths),
                'val_samples': len(val_paths),
                'input_channels': 1
            }
        }
        
        with open(f'benchmark_results_{arch_name}_merged.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        
        notify_macos("Done training", accuracy*100)
        
        return benchmark_results
    else:
        notify_macos("Done training", best_val_acc)
        return {'best_val_accuracy': best_val_acc}

if __name__ == "__main__":
    import sys
    arch_name = sys.argv[1] if len(sys.argv) > 1 else 'mobilenet_v3_small'
    finetune_merged(arch_name=arch_name)

import torch
import torch.nn as nn
import numpy as np
import os
import onnxruntime as ort

class HorizontalEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_c = 1 # Grayscale Highpass only
        # 10 stages: 1024->512->256->128->64->32->16->8->4->2->1
        # Reduced intermediate channels and kernel size for performance on MPS
        channels = [8, 16, 32, 64, 128, 256, 256, 512, 512, 1024]
        for out_c in channels:
            # Using 11 instead of 31 for performance, still "horizontal"
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=(3, 11), stride=(2, 1), padding=(1, 5)))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        try:
            self.load_state_dict(torch.load("horizontal_embedder_ae.pt", map_location='cpu', weights_only=True))
            print("success")
        except:
            print("will start weights from random")

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

def train_xgboost():
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    import pickle
    from scripts.data_loader import get_dataloader

    model = HorizontalEmbedder()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    if os.path.exists("embeddings_cache.pkl"):
        print("Loading cached embeddings...")
        with open("embeddings_cache.pkl", "rb") as f:
            embeddings, labels = pickle.load(f)
    else:
        print("Computing embeddings...")
        loader = get_dataloader(batch_size=1, shuffle=False, return_labels=True)
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch, label in loader:
                batch = batch.to(device)
                emb = model(batch).cpu().numpy()
                embeddings.append(emb)
                labels.extend(label)
        embeddings = np.vstack(embeddings)
        with open("embeddings_cache.pkl", "wb") as f:
            pickle.dump((embeddings, labels), f)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    clf = xgb.XGBClassifier()
    clf.fit(embeddings, y)
    
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump((clf, le), f)
    print("XGBoost training complete.")

def export():
    model = HorizontalEmbedder()
    model.eval()
    dummy_input = torch.randn(1, 1, 1024, 1024)
    onnx_path = "horizontal_embedder.onnx"
    print(f"Exporting to {onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=17,
                          do_constant_folding=True, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    session = ort.InferenceSession(onnx_path)
    output = session.run(None, {'input': dummy_input.numpy()})
    print(f"Export successful. Output shape: {output[0].shape}")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantized_path = "horizontal_embedder_int8.onnx"
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    print(f"Original size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {os.path.getsize(quantized_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    export()
    train_xgboost()

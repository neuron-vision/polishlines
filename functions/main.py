# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
import torch
import timm
from PIL import Image
import io
import base64
import numpy as np
import json

set_global_options(max_instances=10, memory=2048)

initialize_app()

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading HRNet-W48 model...")
        _model = timm.create_model('hrnet_w48', pretrained=True, num_classes=0)
        _model.float()
        _model.eval()
        print("Model loaded.")
    return _model

@https_fn.on_request()
def calc_hrnet_w48_embedder(req: https_fn.Request) -> https_fn.Response:
    if req.method == "OPTIONS":
        return https_fn.Response(status=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
        })
    
    headers = {"Access-Control-Allow-Origin": "*"}
    try:
        data = req.get_json()
        image_b64 = data.get("image")
        if not image_b64:
            return https_fn.Response("Missing image", status=400, headers=headers)
        
        image_data = base64.b64decode(image_b64.split(",")[-1])
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        
        img_arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_arr = (img_arr - mean) / std
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).float()
        
        with torch.no_grad():
            features = get_model()(img_tensor)
            vector = features.squeeze(0).cpu().numpy().tolist()
            
        return https_fn.Response(
            json.dumps({"vector_size": len(vector)}),
            mimetype="application/json",
            headers=headers
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return https_fn.Response(str(e), status=500, headers=headers)

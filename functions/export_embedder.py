import torch
import timm
import os
import numpy as np

def sanity_check(model_path):
    print(f"\n--- Sanity Check: {model_path} ---")
    try:
        import onnxruntime as ort
        
        # Load the model
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Prepare dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        
        print(f"Inference successful!")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output mean: {np.mean(outputs[0]):.4f}")
        return True
    except Exception as e:
        print(f"Sanity check failed: {e}")
        return False

def export():
    print("Loading HRNet-W48...")
    model = timm.create_model('hrnet_w48', pretrained=True, num_classes=0)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "hrnet_w48.onnx"
    
    # Export with latest dynamo-based exporter (requires torch >= 2.1)
    print(f"Exporting to ONNX: {onnx_path}...")
    # Use dynamo=True for the latest support if available, otherwise fallback
    try:
        # Latest torch.onnx.export style with Dynamo
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    except Exception as e:
        print(f"Dynamo export failed, falling back to legacy: {e}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    sanity_check(onnx_path)

    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        print("\nQuantizing to INT8...")
        quantized_path = "hrnet_w48_int8.onnx"
        quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
        
        # Verify sizes
        orig_size = os.path.getsize(onnx_path)
        quant_size = os.path.getsize(quantized_path)
        
        print(f"Quantized model saved to: {quantized_path}")
        print(f"Original size: {orig_size / 1024 / 1024:.2f} MB")
        print(f"Quantized size: {quant_size / 1024 / 1024:.2f} MB")
        
        sanity_check(quantized_path)
        
    except ImportError:
        print("onnxruntime-quantization not installed. Skipping quantization.")

if __name__ == "__main__":
    export()

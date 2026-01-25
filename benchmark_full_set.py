import json
import numpy as np
from pathlib import Path
from benchmark_scan_lines import test_variation

variations = [
    {'name': 'Var1: Edge low threshold (best)', 'kernel_length': 50, 'kernel_thickness': 3, 
     'step': 10, 'min_correlation': 0.05, 'correlation_threshold': 0.08, 'use_edges': True, 'use_fft': False, 'edge_low': 30, 'edge_high': 100},
    {'name': 'Var2: Edge + lower threshold', 'kernel_length': 50, 'kernel_thickness': 3, 
     'step': 10, 'min_correlation': 0.05, 'correlation_threshold': 0.06, 'use_edges': True, 'use_fft': False, 'edge_low': 30, 'edge_high': 100},
    {'name': 'Var3: Edge + fine step', 'kernel_length': 50, 'kernel_thickness': 3, 
     'step': 8, 'min_correlation': 0.05, 'correlation_threshold': 0.08, 'use_edges': True, 'use_fft': False, 'edge_low': 30, 'edge_high': 100},
    {'name': 'Var4: Edge + thicker lines', 'kernel_length': 50, 'kernel_thickness': 4, 
     'step': 10, 'min_correlation': 0.05, 'correlation_threshold': 0.08, 'use_edges': True, 'use_fft': False, 'edge_low': 30, 'edge_high': 100},
    {'name': 'Var5: Edge + larger kernel', 'kernel_length': 60, 'kernel_thickness': 3, 
     'step': 10, 'min_correlation': 0.05, 'correlation_threshold': 0.08, 'use_edges': True, 'use_fft': False, 'edge_low': 30, 'edge_high': 100},
]

if __name__ == "__main__":
    train_dir = Path("data/train")
    preprocessed_dir = Path("preprocessed_scan_lines")
    
    all_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    test_folders = []
    for folder in all_folders:
        extra_data_path = folder / "Extra Data.json"
        if extra_data_path.exists():
            with open(extra_data_path, 'r') as f:
                extra_data = json.load(f)
                label = extra_data.get("Polish Lines Data", {}).get("User Input", "Unknown")
                angles = extra_data.get("Polish Lines Data", {}).get("Chosen Facet PD", [])
                if angles and label in ["Has_PL", "No_PL"]:
                    if (preprocessed_dir / f"{folder.name}.png").exists():
                        test_folders.append(folder)
    
    print(f"Found {len(test_folders)} folders with angles and preprocessed data")
    print(f"Testing all {len(test_folders)} folders...\n")
    
    results = {}
    
    for var in variations:
        print(f"Testing {var['name']}...")
        correct = 0
        total = 0
        errors = []
        has_pl_correct = 0
        has_pl_total = 0
        no_pl_correct = 0
        no_pl_total = 0
        
        for folder in test_folders:
            is_correct, details = test_variation(
                folder,
                kernel_length=var['kernel_length'],
                kernel_thickness=var['kernel_thickness'],
                step=var['step'],
                min_correlation=var['min_correlation'],
                correlation_threshold=var['correlation_threshold'],
                use_edges=var.get('use_edges', False),
                use_fft=var.get('use_fft', False),
                edge_low=var.get('edge_low', 50),
                edge_high=var.get('edge_high', 150)
            )
            
            if details is not None:
                total += 1
                if is_correct:
                    correct += 1
                errors.append(details['angle_error'])
                
                label = details.get('label', 'Unknown')
                if label == "Has_PL":
                    has_pl_total += 1
                    if is_correct:
                        has_pl_correct += 1
                elif label == "No_PL":
                    no_pl_total += 1
                    if is_correct:
                        no_pl_correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        avg_error = np.mean(errors) if errors else 0
        has_pl_accuracy = (has_pl_correct / has_pl_total * 100) if has_pl_total > 0 else 0
        no_pl_accuracy = (no_pl_correct / no_pl_total * 100) if no_pl_total > 0 else 0
        
        results[var['name']] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_error': avg_error,
            'has_pl_accuracy': has_pl_accuracy,
            'has_pl_correct': has_pl_correct,
            'has_pl_total': has_pl_total,
            'no_pl_accuracy': no_pl_accuracy,
            'no_pl_correct': no_pl_correct,
            'no_pl_total': no_pl_total,
            'params': var
        }
        
        print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"  Has_PL Accuracy: {has_pl_accuracy:.2f}% ({has_pl_correct}/{has_pl_total})")
        print(f"  No_PL Accuracy: {no_pl_accuracy:.2f}% ({no_pl_correct}/{no_pl_total})")
        print(f"  Avg angle error: {avg_error:.2f}°\n")
    
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for var_name, var_results in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{var_name}: {var_results['accuracy']:.2f}% accuracy, {var_results['avg_error']:.2f}° avg error")
    
    best_var = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest variation: {best_var[0]} with {best_var[1]['accuracy']:.2f}% accuracy")
    
    with open("benchmark_full_set_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to benchmark_full_set_results.json")

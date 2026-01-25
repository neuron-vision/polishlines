import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from benchmark_scan_lines import test_variation

def count_labels():
    train_dir = Path("data/train")
    has_pl_count = 0
    no_pl_count = 0
    unknown_count = 0
    
    for folder in sorted(train_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        extra_data_path = folder / "Extra Data.json"
        if not extra_data_path.exists():
            continue
        
        with open(extra_data_path, 'r') as f:
            extra_data = json.load(f)
            label = extra_data.get("Polish Lines Data", {}).get("User Input", "Unknown")
        
        if label == "Has_PL":
            has_pl_count += 1
        elif label == "No_PL":
            no_pl_count += 1
        else:
            unknown_count += 1
    
    return has_pl_count, no_pl_count, unknown_count

def generate_confusion_matrix(variation_params):
    train_dir = Path("data/train")
    folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    confusion_matrix = {
        'Has_PL': {'Correct': 0, 'Incorrect': 0},
        'No_PL': {'Correct': 0, 'Incorrect': 0},
        'Unknown': {'Tested': 0, 'Skipped': 0}
    }
    
    results = []
    skipped_no_angles = 0
    skipped_no_preprocessing = 0
    has_pl_tested = 0
    has_pl_correct = 0
    no_pl_tested = 0
    no_pl_correct = 0
    
    for folder in folders:
        extra_data_path = folder / "Extra Data.json"
        if not extra_data_path.exists():
            continue
        
        with open(extra_data_path, 'r') as f:
            extra_data = json.load(f)
            true_label = extra_data.get("Polish Lines Data", {}).get("User Input", "Unknown")
            angles = extra_data.get("Polish Lines Data", {}).get("Chosen Facet PD", [])
        
        if not angles:
            skipped_no_angles += 1
            continue
        
        preprocessed_dir = Path("preprocessed_scan_lines")
        preprocessed_image_path = preprocessed_dir / f"{folder.name}.png"
        if not preprocessed_image_path.exists():
            skipped_no_preprocessing += 1
            continue
        
        if true_label not in ["Has_PL", "No_PL"]:
            continue
        
        is_correct, details = test_variation(
            folder,
            kernel_length=variation_params['kernel_length'],
            kernel_thickness=variation_params['kernel_thickness'],
            step=variation_params['step'],
            min_correlation=variation_params['min_correlation'],
            correlation_threshold=variation_params['correlation_threshold']
        )
        
        if details is not None:
            if true_label == "Has_PL":
                has_pl_tested += 1
                if is_correct:
                    has_pl_correct += 1
                    confusion_matrix['Has_PL']['Correct'] += 1
                else:
                    confusion_matrix['Has_PL']['Incorrect'] += 1
            elif true_label == "No_PL":
                no_pl_tested += 1
                if is_correct:
                    no_pl_correct += 1
                    confusion_matrix['No_PL']['Correct'] += 1
                else:
                    confusion_matrix['No_PL']['Incorrect'] += 1
            
            results.append({
                'folder': folder.name,
                'true_label': true_label,
                'is_correct': bool(is_correct),
                'angle_error': float(details.get('angle_error', 0))
            })
    
    return confusion_matrix, results, skipped_no_angles, skipped_no_preprocessing, has_pl_tested, has_pl_correct, no_pl_tested, no_pl_correct

if __name__ == "__main__":
    print("="*60)
    print("POLISH LINE DETECTION REPORT")
    print("="*60)
    
    has_pl, no_pl, unknown = count_labels()
    total = has_pl + no_pl + unknown
    
    print(f"\nTotal Images in Dataset:")
    print(f"  Has_PL: {has_pl}")
    print(f"  No_PL: {no_pl}")
    print(f"  Unknown: {unknown}")
    print(f"  Total: {total}")
    
    variation_params = {
        'kernel_length': 30,
        'kernel_thickness': 2,
        'step': 10,
        'min_correlation': 0.08,
        'correlation_threshold': 0.2
    }
    
    print(f"\nTesting with Variation 3 parameters (Medium threshold)...")
    cm, results, skipped_no_angles, skipped_no_preprocessing, has_pl_tested, has_pl_correct, no_pl_tested, no_pl_correct = generate_confusion_matrix(variation_params)
    
    if has_pl_tested > 0 or no_pl_tested > 0:
        variation_params = {
            'kernel_length': 25,
            'kernel_thickness': 2,
            'step': 12,
            'min_correlation': 0.08,
            'correlation_threshold': 0.15
        }
        
        print(f"\nTesting with Variation 1 parameters...")
        cm, results, skipped_no_angles, skipped_no_preprocessing, has_pl_tested, has_pl_correct, no_pl_tested, no_pl_correct = generate_confusion_matrix(variation_params)
        
        print(f"  Has_PL folders tested: {has_pl_tested} (correct: {has_pl_correct})")
        print(f"  No_PL folders tested: {no_pl_tested} (correct: {no_pl_correct})")
        print(f"  Folders skipped (no angles): {skipped_no_angles}")
        print(f"  Folders skipped (no preprocessing): {skipped_no_preprocessing}")
        
        print(f"\nConfusion Matrix:")
        print(f"{'True Label':<15} {'Predicted Has_PL':<20} {'Predicted No_PL':<20} {'Total':<15}")
        print("-" * 70)
        
        has_pl_correct_count = cm['Has_PL']['Correct']
        has_pl_incorrect_count = cm['Has_PL']['Incorrect']
        has_pl_total = has_pl_correct_count + has_pl_incorrect_count
        
        no_pl_correct_count = cm['No_PL']['Correct']
        no_pl_incorrect_count = cm['No_PL']['Incorrect']
        no_pl_total = no_pl_correct_count + no_pl_incorrect_count
        
        print(f"{'Has_PL':<15} {has_pl_correct_count:<20} {has_pl_incorrect_count:<20} {has_pl_total:<15}")
        print(f"{'No_PL':<15} {no_pl_incorrect_count:<20} {no_pl_correct_count:<20} {no_pl_total:<15}")
        
        print("-" * 70)
        total_tested = has_pl_total + no_pl_total
        total_correct = has_pl_correct_count + no_pl_correct_count
        print(f"{'Total':<15} {has_pl_correct_count + no_pl_incorrect_count:<20} {has_pl_incorrect_count + no_pl_correct_count:<20} {total_tested:<15}")
        
        overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0
        has_pl_accuracy = (has_pl_correct_count / has_pl_total * 100) if has_pl_total > 0 else 0
        no_pl_accuracy = (no_pl_correct_count / no_pl_total * 100) if no_pl_total > 0 else 0
        
        print(f"\nPerformance Metrics:")
        print(f"  Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_tested})")
        print(f"  Has_PL Accuracy: {has_pl_accuracy:.2f}% ({has_pl_correct_count}/{has_pl_total})")
        print(f"  No_PL Accuracy: {no_pl_accuracy:.2f}% ({no_pl_correct_count}/{no_pl_total})")
        
        report_data = {
            'total_counts': {
                'Has_PL': has_pl,
                'No_PL': no_pl,
                'Unknown': unknown,
                'Total': total
            },
            'confusion_matrix': cm,
            'metrics': {
                'overall_accuracy': overall_accuracy,
                'has_pl_accuracy': has_pl_accuracy,
                'no_pl_accuracy': no_pl_accuracy,
                'has_pl_tested': has_pl_tested,
                'has_pl_correct': has_pl_correct,
                'no_pl_tested': no_pl_tested,
                'no_pl_correct': no_pl_correct
            },
            'skipped': {
                'no_angles': skipped_no_angles,
                'no_preprocessing': skipped_no_preprocessing
            },
            'results': results
        }
    else:
        print("No preprocessed folders found. Run preprocessing first.")
        report_data = {
            'total_counts': {
                'Has_PL': has_pl,
                'No_PL': no_pl,
                'Unknown': unknown,
                'Total': total
            },
            'note': 'No preprocessed data available for testing'
        }
    
    with open("scan_results_report.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed results saved to scan_results_report.json")

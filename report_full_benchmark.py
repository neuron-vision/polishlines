import json
from pathlib import Path

def count_labels():
    train_dir = Path("data/train")
    folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    
    has_pl = 0
    no_pl = 0
    unknown = 0
    
    for folder in folders:
        extra_data_path = folder / "Extra Data.json"
        if extra_data_path.exists():
            with open(extra_data_path, 'r') as f:
                data = json.load(f)
                label = data.get("Polish Lines Data", {}).get("User Input", "Unknown")
                if label == "Has_PL":
                    has_pl += 1
                elif label == "No_PL":
                    no_pl += 1
                else:
                    unknown += 1
    
    return has_pl, no_pl, unknown

if __name__ == "__main__":
    print("=" * 70)
    print("FULL DATASET BENCHMARK REPORT")
    print("=" * 70)
    
    has_pl, no_pl, unknown = count_labels()
    total = has_pl + no_pl + unknown
    
    print(f"\nTotal Images in Dataset:")
    print(f"  Has_PL: {has_pl}")
    print(f"  No_PL: {no_pl}")
    print(f"  Unknown: {unknown}")
    print(f"  Total: {total}")
    
    benchmark_path = Path("benchmark_full_set_results.json")
    if not benchmark_path.exists():
        print("\nBenchmark results not found. Run benchmark_full_set.py first.")
        exit(1)
    
    with open(benchmark_path, 'r') as f:
        results = json.load(f)
    
    print(f"\nBenchmark Results (Full Dataset):")
    print("=" * 70)
    
    best_var = max(results.items(), key=lambda x: x[1]['accuracy'])
    var_name = best_var[0]
    var_results = best_var[1]
    
    print(f"\nBest Variation: {var_name}")
    print(f"  Overall Accuracy: {var_results['accuracy']:.2f}% ({var_results['correct']}/{var_results['total']})")
    print(f"  Has_PL Accuracy: {var_results['has_pl_accuracy']:.2f}% ({var_results['has_pl_correct']}/{var_results['has_pl_total']})")
    print(f"  No_PL Accuracy: {var_results['no_pl_accuracy']:.2f}% ({var_results['no_pl_correct']}/{var_results['no_pl_total']})")
    print(f"  Average Angle Error: {var_results['avg_error']:.2f}°")
    
    print(f"\nAll Variations:")
    print("-" * 70)
    print(f"{'Variation':<50} {'Overall':<12} {'Has_PL':<12} {'No_PL':<12} {'Avg Error':<12}")
    print("-" * 70)
    
    for var_name, var_results in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{var_name:<50} {var_results['accuracy']:>6.2f}%     {var_results['has_pl_accuracy']:>6.2f}%     {var_results['no_pl_accuracy']:>6.2f}%     {var_results['avg_error']:>6.2f}°")
    
    print("\n" + "=" * 70)
    print("Confusion Matrix (Best Variation):")
    print("=" * 70)
    
    has_pl_correct = var_results['has_pl_correct']
    has_pl_incorrect = var_results['has_pl_total'] - has_pl_correct
    no_pl_correct = var_results['no_pl_correct']
    no_pl_incorrect = var_results['no_pl_total'] - no_pl_correct
    
    print(f"{'True Label':<15} {'Predicted Has_PL':<20} {'Predicted No_PL':<20} {'Total':<15}")
    print("-" * 70)
    print(f"{'Has_PL':<15} {has_pl_correct:<20} {has_pl_incorrect:<20} {var_results['has_pl_total']:<15}")
    print(f"{'No_PL':<15} {no_pl_incorrect:<20} {no_pl_correct:<20} {var_results['no_pl_total']:<15}")
    print("-" * 70)
    total_tested = var_results['has_pl_total'] + var_results['no_pl_total']
    total_correct = has_pl_correct + no_pl_correct
    print(f"{'Total':<15} {has_pl_correct + no_pl_incorrect:<20} {has_pl_incorrect + no_pl_correct:<20} {total_tested:<15}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Overall Accuracy: {var_results['accuracy']:.2f}% ({total_correct}/{total_tested})")
    print(f"  Has_PL Precision: {has_pl_correct / (has_pl_correct + no_pl_incorrect) * 100:.2f}%" if (has_pl_correct + no_pl_incorrect) > 0 else "  Has_PL Precision: N/A")
    print(f"  Has_PL Recall: {var_results['has_pl_accuracy']:.2f}%")
    print(f"  No_PL Precision: {no_pl_correct / (no_pl_correct + has_pl_incorrect) * 100:.2f}%" if (no_pl_correct + has_pl_incorrect) > 0 else "  No_PL Precision: N/A")
    print(f"  No_PL Recall: {var_results['no_pl_accuracy']:.2f}%")
    
    report_data = {
        'total_counts': {
            'Has_PL': has_pl,
            'No_PL': no_pl,
            'Unknown': unknown,
            'Total': total
        },
        'best_variation': {
            'name': var_name,
            'results': var_results
        },
        'all_variations': results,
        'confusion_matrix': {
            'Has_PL': {'Correct': has_pl_correct, 'Incorrect': has_pl_incorrect, 'Total': var_results['has_pl_total']},
            'No_PL': {'Correct': no_pl_correct, 'Incorrect': no_pl_incorrect, 'Total': var_results['no_pl_total']}
        }
    }
    
    with open("full_benchmark_report.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to full_benchmark_report.json")

import json
from pathlib import Path
from finetune_mobilenet import finetune_mobilenet

architectures = [
    'mobilenet_v3_small',
    'mobilenet_v3_large',
    'resnet18',
    'resnet34',
    'efficientnet_b0',
    'efficientnet_b1'
]

all_results = []

for arch in architectures:
    print(f"\n{'='*60}")
    print(f"Training {arch}")
    print(f"{'='*60}\n")
    
    results = finetune_mobilenet(arch_name=arch)
    all_results.append({
        'architecture': arch,
        'overall_accuracy': results['overall_accuracy'],
        'best_val_accuracy': results['best_val_accuracy'],
        'per_class_metrics': results['per_class_metrics']
    })

with open('all_architectures_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*60)
print("SUMMARY OF ALL ARCHITECTURES")
print("="*60)
print(f"\n{'Architecture':<25} {'Overall Acc':<15} {'Best Val Acc':<15}")
print("-"*60)
for result in all_results:
    print(f"{result['architecture']:<25} {result['overall_accuracy']*100:>6.2f}%      {result['best_val_accuracy']:>6.2f}%")

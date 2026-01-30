import yaml
from datetime import datetime
from pathlib import Path
from common_utils import ROOT_FOLDER
from run_plan import run_plan

ACCURACY_THRESHOLD = 0.90

def generate_report(results, best_result):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    reports_dir = ROOT_FOLDER / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"run_report_{ts}.md"
    
    m = best_result['metrics']
    lines = [
        f"# Plan Execution Report",
        f"Generated: {datetime.now().isoformat()}",
        f"",
        f"## Best Result",
        f"- Plan: **{best_result['plan_name']}**",
        f"- Notes: {best_result['notes']}",
        f"- Best Accuracy: **{m['best_accuracy']:.2%}**",
        f"- ML Accuracy (LOO-CV): **{m.get('ml_accuracy', 0):.2%}**",
        f"- ML AUC: **{m.get('ml_auc', 0):.4f}**",
        f"- Threshold Accuracy: {m.get('threshold_accuracy', m.get('best_accuracy', 0)):.2%}",
        f"- Best Single Feature: {m.get('best_single_feature', 'N/A')} (AUC={m.get('best_single_feature_auc', 0):.4f})",
        f"",
        f"### Best Plan Parameters",
        f"```yaml",
    ]
    lines.append(yaml.dump(best_result['params'], default_flow_style=False) if best_result['params'] else "# baseline (no overrides)")
    lines.extend([
        f"```",
        f"",
        f"## All Results Summary",
        f"| Plan | Best Acc | ML Acc | ML AUC | Best Feature |",
        f"|------|----------|--------|--------|--------------|",
    ])
    
    for r in sorted(results, key=lambda x: x['metrics'].get('best_accuracy', 0), reverse=True):
        m = r['metrics']
        if 'error' in m:
            lines.append(f"| {r['plan_name']} | ERROR | - | - | - |")
        else:
            lines.append(f"| {r['plan_name']} | {m['best_accuracy']:.2%} | {m.get('ml_accuracy', 0):.2%} | {m.get('ml_auc', 0):.4f} | {m.get('best_single_feature', 'N/A')} |")
    
    lines.extend([
        f"",
        f"## Detailed Results",
    ])
    
    for r in results:
        lines.extend([
            f"",
            f"### {r['plan_name']}",
            f"- Notes: {r['notes']}",
            f"- Params: `{r['params']}`",
        ])
        m = r['metrics']
        if 'error' in m:
            lines.append(f"- **Error**: {m['error']}")
        else:
            lines.extend([
                f"- Best Accuracy: {m['best_accuracy']:.2%}",
                f"- ML Accuracy: {m.get('ml_accuracy', 0):.2%}, ML AUC: {m.get('ml_auc', 0):.4f}",
                f"- Threshold Accuracy: {m.get('threshold_accuracy', 0):.2%}, Threshold AUC: {m.get('threshold_auc', 0):.4f}",
                f"- Best Single Feature: {m.get('best_single_feature', 'N/A')} (AUC={m.get('best_single_feature_auc', 0):.4f})",
                f"- Has_PL (n={m['has_pl_count']}), No_PL (n={m['no_pl_count']})",
            ])
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to {report_path}")
    return report_path

if __name__ == "__main__":
    with open(ROOT_FOLDER / "plans.yml") as f:
        plans = yaml.safe_load(f)['plans']
    
    results = []
    best_result = None
    
    for i, plan in enumerate(plans):
        print(f"\n{'='*60}")
        print(f"Running plan {i+1}/{len(plans)}: {plan['plan_name']}")
        print(f"{'='*60}")
        
        result = run_plan(i)
        results.append(result)
        
        acc = result['metrics'].get('best_accuracy', 0)
        ml_acc = result['metrics'].get('ml_accuracy', 0)
        best_acc = max(acc, ml_acc)
        
        if best_result is None or best_acc > best_result['metrics'].get('best_accuracy', 0):
            best_result = result
        
        if best_acc >= ACCURACY_THRESHOLD:
            print(f"\n*** SUCCESS: {plan['plan_name']} achieved {best_acc:.2%} accuracy >= {ACCURACY_THRESHOLD:.0%} ***")
            generate_report(results, best_result)
            break
    else:
        print(f"\nNo plan achieved {ACCURACY_THRESHOLD:.0%} accuracy")
        print(f"Best result: {best_result['plan_name']} with {best_result['metrics'].get('best_accuracy', 0):.2%}")
        generate_report(results, best_result)

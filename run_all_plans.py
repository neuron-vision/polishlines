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
    
    lines = [
        f"# Plan Execution Report",
        f"Generated: {datetime.now().isoformat()}",
        f"",
        f"## Best Result",
        f"- Plan: **{best_result['plan_name']}**",
        f"- Notes: {best_result['notes']}",
        f"- Accuracy: **{best_result['metrics']['best_accuracy']:.2%}**",
        f"- AUC: **{best_result['metrics']['auc']:.4f}**",
        f"- Threshold: {best_result['metrics']['best_threshold']:.6f}",
        f"- Has_PL above threshold: {best_result['metrics']['has_pl_above_thresh_pct']:.2%}",
        f"- No_PL below threshold: {best_result['metrics']['no_pl_below_thresh_pct']:.2%}",
        f"",
        f"### Best Plan Parameters",
        f"```yaml",
    ]
    lines.append(yaml.dump(best_result['params'], default_flow_style=False) if best_result['params'] else "# baseline (no overrides)")
    lines.extend([
        f"```",
        f"",
        f"## All Results Summary",
        f"| Plan | Accuracy | AUC | Has_PL% | No_PL% |",
        f"|------|----------|-----|---------|--------|",
    ])
    
    for r in sorted(results, key=lambda x: x['metrics'].get('best_accuracy', 0), reverse=True):
        m = r['metrics']
        if 'error' in m:
            lines.append(f"| {r['plan_name']} | ERROR | - | - | - |")
        else:
            lines.append(f"| {r['plan_name']} | {m['best_accuracy']:.2%} | {m['auc']:.4f} | {m['has_pl_above_thresh_pct']:.2%} | {m['no_pl_below_thresh_pct']:.2%} |")
    
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
                f"- Accuracy: {m['best_accuracy']:.2%}",
                f"- AUC: {m['auc']:.4f}",
                f"- Threshold: {m['best_threshold']:.6f}",
                f"- Has_PL (n={m['has_pl_count']}): mean={m['has_pl_mean']:.6f}, median={m['has_pl_median']:.6f}, above_thresh={m['has_pl_above_thresh_pct']:.2%}",
                f"- No_PL (n={m['no_pl_count']}): mean={m['no_pl_mean']:.6f}, median={m['no_pl_median']:.6f}, below_thresh={m['no_pl_below_thresh_pct']:.2%}",
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
        if best_result is None or acc > best_result['metrics'].get('best_accuracy', 0):
            best_result = result
        
        if acc >= ACCURACY_THRESHOLD:
            print(f"\n*** SUCCESS: {plan['plan_name']} achieved {acc:.2%} accuracy >= {ACCURACY_THRESHOLD:.0%} ***")
            generate_report(results, best_result)
            break
    else:
        print(f"\nNo plan achieved {ACCURACY_THRESHOLD:.0%} accuracy")
        print(f"Best result: {best_result['plan_name']} with {best_result['metrics'].get('best_accuracy', 0):.2%}")
        generate_report(results, best_result)

import subprocess
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from common_utils import ROOT_FOLDER, gather_all_data, load_meta_params

def build_param_args(params):
    args = []
    for k, v in params.items():
        if isinstance(v, list):
            # Format list without spaces for shell compatibility
            import json
            args.append(f"{k}={json.dumps(v, separators=(',', ':'))}")
        elif isinstance(v, bool):
            args.append(f"{k}={'true' if v else 'false'}")
        else:
            args.append(f"{k}={v}")
    return args

def calculate_features(all_data):
    assert len(all_data) > 0, "No data found in processed_data folder"
    flat_rows = []
    for row in all_data:
        highest_stds = []
        for rotated_image in row['rotated_images']:
            kernels_data = pd.DataFrame([
                dict(high_pass_std=k.get('high_pass_std', np.std(k['high_pass'])))
                for k in rotated_image['powers']
            ])
            highest_std = kernels_data['high_pass_std'].max()
            highest_stds.append(highest_std)
        highpass_std_max = np.max(highest_stds) if highest_stds else 0
        flat_rows.append(dict(label=row['label'], highpass_std_max=highpass_std_max))
    return pd.DataFrame(flat_rows)

def compute_separation_metrics(df):
    has_pl = df[df['label'] == 'Has_PL']['highpass_std_max'].values
    no_pl = df[df['label'] == 'No_PL']['highpass_std_max'].values
    
    if len(has_pl) == 0 or len(no_pl) == 0:
        return dict(error="missing group data")
    
    y_true = np.concatenate([np.ones(len(has_pl)), np.zeros(len(no_pl))])
    y_scores = np.concatenate([has_pl, no_pl])
    auc = roc_auc_score(y_true, y_scores)
    
    all_vals = np.concatenate([has_pl, no_pl])
    best_acc, best_thresh = 0, 0
    for thresh in np.percentile(all_vals, np.arange(1, 100)):
        acc = (np.sum(has_pl >= thresh) + np.sum(no_pl < thresh)) / len(all_vals)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    
    has_pl_above = np.sum(has_pl >= best_thresh) / len(has_pl)
    no_pl_below = np.sum(no_pl < best_thresh) / len(no_pl)
    
    return dict(
        auc=float(auc),
        best_accuracy=float(best_acc),
        best_threshold=float(best_thresh),
        has_pl_above_thresh_pct=float(has_pl_above),
        no_pl_below_thresh_pct=float(no_pl_below),
        has_pl_mean=float(np.mean(has_pl)),
        has_pl_median=float(np.median(has_pl)),
        no_pl_mean=float(np.mean(no_pl)),
        no_pl_median=float(np.median(no_pl)),
        has_pl_count=int(len(has_pl)),
        no_pl_count=int(len(no_pl)),
    )

def run_plan(plan_index):
    with open(ROOT_FOLDER / "plans.yml") as f:
        plans = yaml.safe_load(f)['plans']
    
    assert 0 <= plan_index < len(plans), f"plan_index {plan_index} out of range [0, {len(plans)})"
    plan = plans[plan_index]
    plan_name = plan['plan_name']
    notes = plan.get('notes', '')
    params = plan.get('params', {})
    
    print(f"Running plan {plan_index}: {plan_name}")
    print(f"Notes: {notes}")
    print(f"Params: {params}")
    
    param_args = build_param_args(params)
    param_args.append("should_plot_png=false")
    
    cmd = ["bash", "preprocess_all_scan_lines.sh"] + param_args
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT_FOLDER)
    
    all_data = gather_all_data()
    df = calculate_features(all_data)
    metrics = compute_separation_metrics(df)
    
    result = dict(
        plan_index=plan_index,
        plan_name=plan_name,
        notes=notes,
        params=params,
        metrics=metrics,
    )
    
    results_dir = ROOT_FOLDER / "plan_results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / f"{plan_name}.yml", "w") as f:
        yaml.dump(result, f, default_flow_style=False)
    
    print(f"Results saved to plan_results/{plan_name}.yml")
    print(f"Metrics: {metrics}")
    return result

if __name__ == "__main__":
    plan_index = int(sys.argv[1])
    run_plan(plan_index)

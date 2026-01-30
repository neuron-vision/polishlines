import subprocess
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
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
        all_stds, all_maxs, all_ranges, all_means, all_energies, all_peaks = [], [], [], [], [], []
        all_fft_energy, all_fft_peak_mag, all_periodicity = [], [], []
        all_gabor, all_sobel_x, all_sobel_y, all_edge_ratio = [], [], [], []
        # Per-angle best features
        angle_best_energy = []
        angle_best_std = []
        for rotated_image in row['rotated_images']:
            angle_energies = []
            angle_stds = []
            for k in rotated_image['powers']:
                hp = np.array(k['high_pass'])
                std_val = k.get('high_pass_std', np.std(hp))
                energy_val = k.get('high_pass_energy', np.sum(hp**2))
                all_stds.append(std_val)
                all_maxs.append(k.get('high_pass_max', np.max(hp)))
                all_ranges.append(k.get('high_pass_range', np.max(hp) - np.min(hp)))
                all_means.append(k.get('high_pass_mean', np.mean(hp)))
                all_energies.append(energy_val)
                hp_mean = np.mean(hp)
                peaks = k.get('high_pass_peaks', np.sum((hp[1:-1] > hp[:-2]) & (hp[1:-1] > hp[2:]) & (hp[1:-1] > hp_mean + 0.1)))
                all_peaks.append(peaks)
                all_fft_energy.append(k.get('fft_energy', 0))
                all_fft_peak_mag.append(k.get('fft_peak_mag', 0))
                all_periodicity.append(k.get('periodicity', 0))
                angle_energies.append(energy_val)
                angle_stds.append(std_val)
            angle_best_energy.append(np.max(angle_energies) if angle_energies else 0)
            angle_best_std.append(np.max(angle_stds) if angle_stds else 0)
            # Texture features
            gabor = rotated_image.get('gabor_responses', [0])
            all_gabor.extend(gabor)
            all_sobel_x.append(rotated_image.get('sobel_x_energy', 0))
            all_sobel_y.append(rotated_image.get('sobel_y_energy', 0))
            all_edge_ratio.append(rotated_image.get('edge_ratio', 1))
        
        # Variance across angles/kernels
        std_variance = np.var(all_stds) if len(all_stds) > 1 else 0
        energy_variance = np.var(all_energies) if len(all_energies) > 1 else 0
        # Ratio: best angle vs mean of other angles
        if len(angle_best_energy) > 1:
            best_idx = np.argmax(angle_best_energy)
            other_mean = np.mean([e for i, e in enumerate(angle_best_energy) if i != best_idx])
            energy_ratio = angle_best_energy[best_idx] / (other_mean + 1e-10)
        else:
            energy_ratio = 1.0
        flat_rows.append(dict(
            label=row['label'],
            highpass_std_max=np.max(all_stds) if all_stds else 0,
            highpass_std_mean=np.mean(all_stds) if all_stds else 0,
            highpass_std_var=std_variance,
            highpass_max_max=np.max(all_maxs) if all_maxs else 0,
            highpass_range_max=np.max(all_ranges) if all_ranges else 0,
            highpass_range_mean=np.mean(all_ranges) if all_ranges else 0,
            highpass_energy_max=np.max(all_energies) if all_energies else 0,
            highpass_energy_mean=np.mean(all_energies) if all_energies else 0,
            highpass_energy_var=energy_variance,
            highpass_energy_ratio=energy_ratio,
            highpass_peaks_max=np.max(all_peaks) if all_peaks else 0,
            highpass_peaks_sum=np.sum(all_peaks) if all_peaks else 0,
            fft_energy_max=np.max(all_fft_energy) if all_fft_energy else 0,
            fft_energy_mean=np.mean(all_fft_energy) if all_fft_energy else 0,
            fft_peak_mag_max=np.max(all_fft_peak_mag) if all_fft_peak_mag else 0,
            periodicity_max=np.max(all_periodicity) if all_periodicity else 0,
            periodicity_mean=np.mean(all_periodicity) if all_periodicity else 0,
            gabor_max=np.max(all_gabor) if all_gabor else 0,
            gabor_mean=np.mean(all_gabor) if all_gabor else 0,
            sobel_x_max=np.max(all_sobel_x) if all_sobel_x else 0,
            sobel_y_max=np.max(all_sobel_y) if all_sobel_y else 0,
            edge_ratio_max=np.max(all_edge_ratio) if all_edge_ratio else 0,
            edge_ratio_mean=np.mean(all_edge_ratio) if all_edge_ratio else 0,
        ))
    return pd.DataFrame(flat_rows)

FEATURE_COLS = ['highpass_std_max', 'highpass_std_mean', 'highpass_std_var', 'highpass_max_max', 
                'highpass_range_max', 'highpass_range_mean', 'highpass_energy_max',
                'highpass_energy_mean', 'highpass_energy_var', 'highpass_energy_ratio', 'highpass_peaks_max', 'highpass_peaks_sum',
                'fft_energy_max', 'fft_energy_mean', 'fft_peak_mag_max', 'periodicity_max', 'periodicity_mean',
                'gabor_max', 'gabor_mean', 'sobel_x_max', 'sobel_y_max', 'edge_ratio_max', 'edge_ratio_mean']

def compute_separation_metrics(df):
    has_pl_df = df[df['label'] == 'Has_PL']
    no_pl_df = df[df['label'] == 'No_PL']
    
    if len(has_pl_df) == 0 or len(no_pl_df) == 0:
        return dict(error="missing group data")
    
    y = (df['label'] == 'Has_PL').astype(int).values
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    
    # Single feature threshold-based (legacy)
    has_pl = has_pl_df['highpass_std_max'].values
    no_pl = no_pl_df['highpass_std_max'].values
    y_scores_single = df['highpass_std_max'].values
    auc_single = roc_auc_score(y, y_scores_single)
    
    all_vals = np.concatenate([has_pl, no_pl])
    best_acc_single, best_thresh = 0, 0
    for thresh in np.percentile(all_vals, np.arange(1, 100)):
        acc = (np.sum(has_pl >= thresh) + np.sum(no_pl < thresh)) / len(all_vals)
        if acc > best_acc_single:
            best_acc_single, best_thresh = acc, thresh
    
    # Find top 5 features by individual AUC
    feature_aucs = [(col, roc_auc_score(y, df[col].values)) for col in feature_cols]
    feature_aucs.sort(key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in feature_aucs[:5]]
    X_top = df[top_features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)
    
    # Try both logistic regression and gradient boosting
    best_ml_acc, best_ml_auc, best_ml_probs = 0, 0, None
    for clf_class, clf_params in [
        (LogisticRegression, dict(max_iter=1000, C=1.0)),
        (GradientBoostingClassifier, dict(n_estimators=50, max_depth=2, random_state=42)),
    ]:
        preds = np.zeros(len(y))
        probs = np.zeros(len(y))
        for i in range(len(y)):
            X_train = np.delete(X_scaled, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X_scaled[i:i+1]
            clf = clf_class(**clf_params)
            clf.fit(X_train, y_train)
            preds[i] = clf.predict(X_test)[0]
            probs[i] = clf.predict_proba(X_test)[0, 1]
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, probs)
        if acc > best_ml_acc:
            best_ml_acc, best_ml_auc, best_ml_probs = acc, auc, probs
    
    ml_accuracy = best_ml_acc
    ml_auc = best_ml_auc
    
    # Best single feature
    best_feat, best_feat_auc = 'highpass_std_max', auc_single
    for col in feature_cols:
        feat_auc = roc_auc_score(y, df[col].values)
        if feat_auc > best_feat_auc:
            best_feat, best_feat_auc = col, feat_auc
    
    best_accuracy = max(best_acc_single, ml_accuracy)
    
    return dict(
        best_accuracy=float(best_accuracy),
        threshold_accuracy=float(best_acc_single),
        threshold_auc=float(auc_single),
        best_threshold=float(best_thresh),
        ml_accuracy=float(ml_accuracy),
        ml_auc=float(ml_auc),
        best_single_feature=best_feat,
        best_single_feature_auc=float(best_feat_auc),
        has_pl_above_thresh_pct=float(np.sum(has_pl >= best_thresh) / len(has_pl)),
        no_pl_below_thresh_pct=float(np.sum(no_pl < best_thresh) / len(no_pl)),
        has_pl_count=int(len(has_pl_df)),
        no_pl_count=int(len(no_pl_df)),
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

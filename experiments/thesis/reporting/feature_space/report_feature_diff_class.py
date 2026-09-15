import json
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from config.helpers import load_json, extract_class_metrics
from config.constants import ALL_CLASSIFIERS

def version_domain_gap(force=True):
    results_path = Path("/home/stud/nemmler/retristyle/results/domain_gap/feature_space")
    pred_base_path = Path("/home/stud/nemmler/retristyle/results/style_check/styleid")

    output_dir = results_path / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results_flat.json"
    csv_path = output_dir / "results_flat.csv" 
    
    if (json_path.exists() and csv_path.exists()) and not force:
        return pd.read_csv(csv_path, dtype={'class_id': str})

    base = "test_r"
    results = []

    for cl in ALL_CLASSIFIERS:
        cl_name = f"{cl}_domain_gap.json"
        base_data = load_json(results_path / base / cl_name)
        base_classes = base_data["domain_gap"]["per_class"]

        for i in range(1, 5):
            k = f"{base}_k{i}"
            
            k_data = load_json(results_path / k / cl_name)
            k_classes = k_data["domain_gap"]["per_class"]

            pred_fn = f"{cl}_geometric_vanilla_nviews1_seed71397589.json"
            pred_file = pred_base_path / f"view_{i:03d}.png" / "tta_inference" / "predictions" / "imagenet" / "test_r" / pred_fn
            pred_metrics = extract_class_metrics(pred_file)

            for class_id, metrics in k_classes.items():
                class_preds = pred_metrics.get(class_id, {})

                for metric_name, value in metrics.items():
                    
                    if metric_name == "n_samples":
                        continue

                    base_val = base_classes[class_id][metric_name]
                    delta = value - base_val

                    results.append({
                        "classifier": cl,
                        "k_split": k,
                        "k_idx": i,
                        "class_id": str(class_id),
                        "metric": metric_name,
                        "delta": delta,
                        "k_val": value,
                        "base_val": base_val,
                        "acc": class_preds.get("acc"),
                        "conf": class_preds.get("conf"),
                        "ent": class_preds.get("ent")
                    })

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    return df


def calculate_correlations(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Calculates correlations between domain gap metrics and prediction metrics."""
    pred_cols = ["acc", "conf", "ent"]
    gap_cols = ["base_val", "delta", "k_val"]
    
    correlations = []
    
    for (cl, metric), group in df.groupby(["classifier", "metric"]):
        for gap_col in gap_cols:
            for pred_col in pred_cols:
                valid_data = group[[gap_col, pred_col]].dropna()
                
                corr_val = valid_data[gap_col].corr(valid_data[pred_col], method=method) if len(valid_data) > 1 else None
                
                correlations.append({
                    "classifier": cl,
                    "gap_metric": metric,
                    "gap_type": gap_col,
                    "pred_metric": pred_col,
                    f"{method}_corr": corr_val
                })
                
    return pd.DataFrame(correlations)


def print_correlation_table(df: pd.DataFrame, target_k: str = "test_r_k1", method: str = "spearman"):
    """Prints pivot tables comparing domain gap metrics against baseline predictions."""
    sub_df = df[df["k_split"] == target_k].copy()
    
    correlations = []
    for (cl, metric), group in sub_df.groupby(["classifier", "metric"]):
        row = {"classifier": cl, "metric": metric}
        for pred_col in ["acc", "conf", "ent"]:
            for gap_col in ["base_val", "delta", "k_val"]:
                valid = group[[gap_col, pred_col]].dropna()
                corr = valid[gap_col].corr(valid[pred_col], method=method) if len(valid) > 1 else None
                row[f"{gap_col}_{pred_col}"] = corr
        correlations.append(row)

    corr_df = pd.DataFrame(correlations)
    
    print(f"=== Correlation Matrix for {target_k} ({method.capitalize()}) ===")
    print("\nBaseline Domain Gap (base_val) vs Prediction Metrics:")
    print(corr_df.pivot(index="classifier", columns="metric", values=["base_val_acc", "base_val_conf", "base_val_ent"]).to_string())

    print("\nDelta Domain Gap (delta) vs Prediction Metrics:")
    print(corr_df.pivot(index="classifier", columns="metric", values=["delta_acc", "delta_conf", "delta_ent"]).to_string())

    print("\nStylized Split Domain Gap (k_val) vs Prediction Metrics:")
    print(corr_df.pivot(index="classifier", columns="metric", values=["k_val_acc", "k_val_conf", "k_val_ent"]).to_string())
    
    return corr_df


def compute_gains_and_correlations(base_path, tta_path, sty_path, gap_df, classifier_name):
    base_class_preds = extract_class_metrics(base_path)
    tta_class_preds = extract_class_metrics(tta_path)
    sty_class_preds = extract_class_metrics(sty_path)
        
    records = []
    all_classes = set(base_class_preds.keys()).intersection(tta_class_preds.keys())
    
    for cls_id in all_classes:
        acc_base = base_class_preds[cls_id].get("acc", 0.0)
        acc_ens = tta_class_preds[cls_id].get("acc", 0.0)
        acc_style = sty_class_preds[cls_id].get("acc", 0.0)
        
        records.append({
            'class_id': str(cls_id),
            'acc_base': acc_base,
            'acc_ens': acc_ens,
            'acc_style': acc_style,
            'gain_ens': acc_ens - acc_base,
            'gain_style': acc_style - acc_base
        })
        
    df_results = pd.DataFrame(records)
    
    # Filter gap_df for current classifier
    cl_gap = gap_df[gap_df['classifier'] == classifier_name].copy()
    cl_gap['class_id'] = cl_gap['class_id'].astype(str)
    
    # Merge prediction gains with feature space metrics
    merged_df = pd.merge(df_results, cl_gap, on='class_id')
    
    print(f"\n--- Performance Gain vs. Baseline Domain Gap: {classifier_name} ---")
    
    # Group by feature distance metric (MMD, Wasserstein, KL, etc.)
    for m_name, group in merged_df.groupby('metric'):
        # Deduplicate to evaluate unique class-level baseline domain gaps (base_val)
        class_level_data = group.drop_duplicates(subset=['class_id']).dropna(subset=['base_val', 'gain_ens', 'gain_style'])
        
        if len(class_level_data) > 1:
            # 1. Gain vs Baseline Domain Gap (d_c^base)
            rho_base_ens, p_base_ens = spearmanr(class_level_data['base_val'], class_level_data['gain_ens'])
            rho_base_style, p_base_style = spearmanr(class_level_data['base_val'], class_level_data['gain_style'])
            
            # 2. Gain vs Delta Domain Gap (delta)
            rho_delta_ens, p_delta_ens = spearmanr(group['delta'], group['gain_ens'])
            rho_delta_style, p_delta_style = spearmanr(group['delta'], group['gain_style'])
            
            print(f"[{m_name}] Baseline Gap (base_val) vs Gain (Ens):   rho = {rho_base_ens:.3f} (p = {p_base_ens:.3e})")
            print(f"[{m_name}] Baseline Gap (base_val) vs Gain (Style): rho = {rho_base_style:.3f} (p = {p_base_style:.3e})")
            print(f"[{m_name}] Delta Gap    (delta)    vs Gain (Ens):   rho = {rho_delta_ens:.3f} (p = {p_delta_ens:.3e})")
            print(f"[{m_name}] Delta Gap    (delta)    vs Gain (Style): rho = {rho_delta_style:.3f} (p = {p_delta_style:.3e})")
            print("-" * 55)

    return merged_df


def print_gains_and_correlations(df):
    base = "/home/stud/nemmler/retristyle/results/baseline/tta_inference/predictions/imagenet/test_r/{cl}_geometric_vanilla_nviews1_seed71397589.json"
    tta = "/home/stud/nemmler/retristyle/results/ablation/retristyle/tta_inference/predictions/imagenet/test_r/{cl}_retristyle_vanilla_dino_nrefs2_seed71397589.json"
    sty = "/home/stud/nemmler/retristyle/results/style_check/styleid/view_001.png/tta_inference/predictions/imagenet/test_r/{cl}_geometric_vanilla_nviews1_seed71397589.json"
    
    for cl in ALL_CLASSIFIERS:
        base_path = Path(base.format(cl=cl))
        tta_path = Path(tta.format(cl=cl))
        sty_path = Path(sty.format(cl=cl))
        
        if base_path.exists() and tta_path.exists() and sty_path.exists():
            _ = compute_gains_and_correlations(base_path, tta_path, sty_path, df, cl)
        else:
            print(f"Skipping {cl}: file paths not found.")


if __name__ == "__main__":
    print("=" * 60)
    print("1. Aggregating Domain Gap & Baseline Prediction Data...")
    print("=" * 60)
    df = version_domain_gap(force=False)
    print(f"Loaded DataFrame with shape: {df.shape}\n")

    print("=" * 60)
    print("2. Summary Correlations (Domain Gap vs. Baseline Performance)")
    print("=" * 60)
    corr_df = calculate_correlations(df, method="spearman")
    print(corr_df.head(12).to_string(index=False))
    print("\n")

    print("=" * 60)
    print("3. Correlation Matrix for Target Split (test_r_k1)")
    print("=" * 60)
    _ = print_correlation_table(df, target_k="test_r_k1", method="spearman")
    print("\n")

    print("=" * 60)
    print("4. Testing Hypothesis: d_c vs. STTTA Gain (Ens & Style-Only)")
    print("=" * 60)
    print_gains_and_correlations(df)
    print("=" * 60)

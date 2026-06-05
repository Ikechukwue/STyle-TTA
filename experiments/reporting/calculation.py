import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from tqdm import tqdm
from experiments.reporting.helpers.helpers import load_json

def extract_class_metrics(pred_json):
    """Extracts accuracy, confidence, and entropy per class from raw predictions."""
    class_data = {}
    for p in pred_json["predictions"]:
        c_id = str(p["y_true"])
        y_pred_probs = np.array(p["y_pred"])
        
        is_correct = int(np.argmax(y_pred_probs) == p["y_true"])
        confidence = np.max(y_pred_probs)
        pred_entropy = entropy(y_pred_probs)
        
        if c_id not in class_data:
            class_data[c_id] = {"acc": [], "conf": [], "ent": []}
            
        class_data[c_id]["acc"].append(is_correct)
        class_data[c_id]["conf"].append(confidence)
        class_data[c_id]["ent"].append(pred_entropy)
        
    # Average across instances to get clean per-class baseline/TTA values
    return {
        c: {
            "acc": np.mean(v["acc"]),
            "conf": np.mean(v["conf"]),
            "ent": np.mean(v["ent"])
        }
        for c, v in class_data.items()
    }

if __name__ == "__main__":
    # Paths setup
    domain_path = "/home/stud/nemmler/retristyle/results/domain_stats/imagenet_test_r/0_30_imagenet_test_r.json"
    baseline_dir = "/home/stud/nemmler/retristyle/results/baseline_1/test_r/tta_inference/predictions/imagenet/test_r/"
    tta_dir = "results/baseline/test_r/tta_inference/predictions/imagenet/test_r/"

    # Load Domain Discrepancies
    domain_json = load_json(domain_path)["class_summaries"]
    
    # Find all unique experimental TTA files
    tta_files = glob.glob(os.path.join(tta_dir, "*_geometric_tpt_*.json"))
    
    correlation_records = []

    print(f"Analyzing TTA improvements over baselines across {len(tta_files)} configurations...")
    for tta_path in tqdm(tta_files, desc="Processing TTA Shifts"):
        tta_filename = os.path.basename(tta_path)
        
        # Extract metadata from TTA filename
        match = re.search(r"([a-zA-Z0-9_]+)_geometric_(tpt)_nviews(\d+)_seed(\d+)", tta_filename)
        if not match:
            continue
        classifier, eval_strat, views, tta_seed = match.groups()
        
        # Locate corresponding baseline file for this specific classifier
        # Baseline uses: vanilla, nviews1, seed265017005
        baseline_pattern = os.path.join(baseline_dir, f"{classifier}_geometric_vanilla_nviews1_seed*.json")
        baseline_match = glob.glob(baseline_pattern)
        
        if not baseline_match:
            print(f"⚠️ Missing baseline for {classifier}, skipping configuration.")
            continue
            
        # Load JSON data structures
        tta_metrics = extract_class_metrics(load_json(tta_path))
        baseline_metrics = extract_class_metrics(load_json(baseline_match[0]))
        
        # Compute Deltas (Improvements) per class
        analysis_rows = []
        for c_id in domain_json.keys():
            if c_id in tta_metrics and c_id in baseline_metrics:
                dom = domain_json[c_id]
                
                # Delta calculations (TTA - Baseline)
                # Note: For entropy, a negative delta means TTA successfully reduced model confusion
                row = {
                    "class_id": int(c_id),
                    "delta_accuracy": tta_metrics[c_id]["acc"] - baseline_metrics[c_id]["acc"],
                    "delta_confidence": tta_metrics[c_id]["conf"] - baseline_metrics[c_id]["conf"],
                    "delta_entropy": tta_metrics[c_id]["ent"] - baseline_metrics[c_id]["ent"],
                    
                    # Core Domain Stats
                    "edge_similarity": dom.get("edge_similarity_mean"),
                    "ldc_haus": dom.get("ldc_haus_mean"),
                    "lpips": dom.get("lpips_score_mean"),
                    "depth_mae": dom.get("depthanything_v2_large_mae_mean")
                }
                analysis_rows.append(row)
                
        # Transition target configurations to temporary DataFrame for immediate calculation
        df_temp = pd.DataFrame(analysis_rows)
        if df_temp.empty:
            continue
            
        # Calculate Spearman Rho correlations between domain features and TTA improvements
        for domain_stat in ["edge_similarity", "ldc_haus", "lpips", "depth_mae"]:
            for target_delta in ["delta_accuracy", "delta_confidence", "delta_entropy"]:
                rho, p_val = spearmanr(df_temp[domain_stat], df_temp[target_delta], nan_policy='omit')
                
                correlation_records.append({
                    "classifier": classifier,
                    "arch_type": "CNN" if "resnet" in classifier or "dense" in classifier else "ViT",
                    "views": int(views),
                    "tta_seed": int(tta_seed),
                    "domain_metric": domain_stat,
                    "delta_metric": target_delta,
                    "spearman_rho": rho,
                    "p_value": p_val
                })

    # Wrap results in a clean Summary Table
    df_correlations = pd.DataFrame(correlation_records)
    
    # Save ONLY the high-level correlation summary table 
    df_correlations.to_csv("tta_delta_domain_correlations.csv", index=False)
    print("\n✅ Correlation Summary Table Generated! Saved to: tta_delta_domain_correlations.csv")

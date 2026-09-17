import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn
from code.config.constants import ALL_CLASSIFIERS
from collections import Counter
# ==========================================
# File Paths & Config
# ==========================================
PATH_BASE_PREDS = "/home/stud/nemmler/style_tta/results/geometric_tta/tta_inference/predictions/imagenet/test_r/{cl}_geometric_vanilla_nviews1_seed71397589.json"
PATH_TTA_PREDS = "/home/stud/nemmler/style_tta/results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/{cl}_adain_tta_zero_dino_nrefs64_seed71397589.json"
PATH_RETRIEVAL = "/home/stud/nemmler/style_tta/results/retrieval_mapping/imagenet/retrieval_mapping_dino_test_r_s71397589.json"
PATH_CLASS_INDEX = "/home/stud/nemmler/style_tta/data/imagenet/imagenetr/imagenet_class_index.json"
PATH_DOMAIN_GAP = "/home/stud/nemmler/style_tta/results/domain_gap/feature_space/aggregated/results_flat.csv"



# Define 6 taxonomic similarity bins
BIN_EDGES = [-0.01, 0.2, 0.4, 0.6, 0.8, 0.99, 1.01]
BIN_LABELS = [
    "Very Low (0.0-0.2)",
    "Low (0.2-0.4)",
    "Medium (0.4-0.6)",
    "High (0.6-0.8)",
    "Very High (0.8-1.0)",
    "Exact Match (1.0)"
]

# ==========================================
# WordNet & Helper Functions
# ==========================================
def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def synset_from_wnid(wnid: str):
    offset = int(wnid[1:])
    pos = wnid[0]
    return wn.synset_from_pos_and_offset(pos, offset)

def get_wup_similarity(syn1, syn2) -> float:
    if syn1 is None or syn2 is None:
        return 0.0
    sim = syn1.wup_similarity(syn2)
    return float(sim) if sim is not None else 0.0

def load_class_synsets(path: str) -> Dict[int, object]:
    class_index = load_json(path)
    id_to_synset = {}
    for idx_str, val in class_index.items():
        wnid = val[0]
        try:
            id_to_synset[int(idx_str)] = synset_from_wnid(wnid)
        except Exception:
            id_to_synset[int(idx_str)] = None
    return id_to_synset

# ==========================================
# Sample-Level Metric Extraction
# ==========================================
def process_sample_level_data(cl: str, id_to_synset: Dict, retrieval_data: Dict) -> pd.DataFrame:
    base_file = Path(PATH_BASE_PREDS.format(cl=cl))
    tta_file = Path(PATH_TTA_PREDS.format(cl=cl))

    if not base_file.exists() or not tta_file.exists():
        print(f"Skipping {cl}: Prediction files not found.")
        return pd.DataFrame()

    base_preds = load_json(str(base_file))["predictions"]
    tta_preds = load_json(str(tta_file))["predictions"]

    records = []
    for idx, (b_sample, t_sample) in enumerate(zip(base_preds, tta_preds)):
        sample_key = str(idx)
        y_true = b_sample["y_true"]
        
        # Get baseline and TTA correctness
        b_correct = int(np.argmax(b_sample["y_pred"]) == y_true)
        t_correct = int(np.argmax(t_sample["y_pred"]) == y_true)
        acc_gain = (t_correct - b_correct) * 100.0

        # Variance across output prediction probabilities
        b_var = float(np.var(b_sample["y_pred"]))
        t_var = float(np.var(t_sample["y_pred"]))

        # Sample-level accuracy variance (if predictions contain multiple views/runs per sample)
        if isinstance(t_sample["y_pred"], list) and len(t_sample["y_pred"]) > 1:
            sample_accs = [int(np.argmax(p) == y_true) * 100.0 for p in t_sample["y_pred"]]
            acc_var = float(np.var(sample_accs))
        else:
            acc_var = 0.0

        if sample_key in retrieval_data and retrieval_data[sample_key]:
            # Dominant class
            retrieved_classes = [item[1] for item in retrieval_data[sample_key]]
            retrieved_class = Counter(retrieved_classes).most_common(1)[0][0]
            # Retrieve top-1 reference class
            #retrieved_class = retrieval_data[sample_key][0][1]
        else:
            retrieved_class = -1

        is_same_class = int(y_true == retrieved_class)

        # WordNet Similarity
        gt_syn = id_to_synset.get(y_true)
        ret_syn = id_to_synset.get(retrieved_class)
        wup_sim = get_wup_similarity(gt_syn, ret_syn)

        # Map similarity to one of the 6 bins
        sim_bin = pd.cut([wup_sim], bins=BIN_EDGES, labels=BIN_LABELS)[0]

        records.append({
            "classifier": cl,
            "sample_idx": idx,
            "class_id": str(y_true),
            "retrieved_class": str(retrieved_class),
            "is_same_class": is_same_class,
            "wup_sim": wup_sim,
            "sim_bin": sim_bin,
            "base_acc": b_correct * 100.0,
            "tta_acc": t_correct * 100.0,
            "acc_gain": acc_gain,
            "base_var": b_var,
            "tta_var": t_var,
            "acc_gain_var": acc_var
        })

    return pd.DataFrame(records)

# ==========================================
# Main Execution Strategy
# ==========================================
def main():
    print("Loading Class Mappings and Retrievals...")
    id_to_synset = load_class_synsets(PATH_CLASS_INDEX)
    retrieval_data = load_json(PATH_RETRIEVAL)
    
    gap_df_exists = Path(PATH_DOMAIN_GAP).exists()
    if gap_df_exists:
        domain_gap_df = pd.read_csv(PATH_DOMAIN_GAP, dtype={"class_id": str})
    else:
        print(f"Warning: {PATH_DOMAIN_GAP} not found. Delta gap metrics will be omitted.")
        domain_gap_df = pd.DataFrame()

    all_samples = []
    for cl in ALL_CLASSIFIERS:
        df_cl = process_sample_level_data(cl, id_to_synset, retrieval_data)
        if not df_cl.empty:
            all_samples.append(df_cl)

    if not all_samples:
        print("No sample data processed.")
        return

    full_df = pd.concat(all_samples, ignore_index=True)
    # ---------------------------------------------------------
    # Analysis 1: Correct-class vs. Incorrect-class Retrieval (with 6 Bins)
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS 1: Same-Class vs. Different-Class Retrieval & 6-Bin Breakdown")
    print("="*80)

    # Breakdown: Same vs Different Class
    same_diff_summary = full_df.groupby(["classifier", "is_same_class"]).agg(
        sample_count=("sample_idx", "count"),
        mean_base_acc=("base_acc", "mean"),
        var_base_acc=("base_acc", "var"),
        mean_tta_acc=("tta_acc", "mean"),
        var_tta_acc=("tta_acc", "var"),
        mean_gain=("acc_gain", "mean"),
        var_gain=("acc_gain", "var"),
        mean_wup_sim=("wup_sim", "mean"),
        var_wup_sim=("wup_sim", "var")
    ).reset_index()

    same_diff_summary["is_same_class"] = same_diff_summary["is_same_class"].map({1: "Same Class", 0: "Different Class"})
    print("\n--- Summary by Match Type ---")
    print(same_diff_summary.to_string(index=False))

    # Breakdown across 6 Taxonomic Bins
    bin_summary = full_df.groupby(["classifier", "sim_bin"], observed=False).agg(
        sample_count=("sample_idx", "count"),
        mean_base_acc=("base_acc", "mean"),
        var_base_acc=("base_acc", "var"),
        mean_tta_acc=("tta_acc", "mean"),
        var_tta_acc=("tta_acc", "var"),
        mean_gain=("acc_gain", "mean"),
        var_gain=("acc_gain", "var")
    ).reset_index()

    print("\n--- Breakdown across 6 Taxonomic Similarity Bins ---")
    print(bin_summary.to_string(index=False))

    # ---------------------------------------------------------
    # Analysis 2: Retrieval Similarity vs. Gain Correlations
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS 2: Retrieval Similarity vs. Gain Correlations")
    print("="*80)

    correlations = []
    for cl, group in full_df.groupby("classifier"):
        # Sample-level Spearman rank correlation
        rho_sample, p_sample = spearmanr(group["wup_sim"], group["acc_gain"])

        # Class-level aggregated correlation (including variance metrics)
        cls_group = group.groupby("class_id").agg(
            mean_wup=("wup_sim", "mean"),
            var_wup=("wup_sim", "var"),
            mean_gain=("acc_gain", "mean"),
            var_gain=("acc_gain", "var")
        ).reset_index()

        # Merge class-level domain gap if available
        if not domain_gap_df.empty:
            cl_gaps = domain_gap_df[domain_gap_df["classifier"] == cl]
            cls_group = pd.merge(cls_group, cl_gaps, on="class_id", how="inner")
            
            for m_name, sub_m in cls_group.groupby("metric"):
                rho_delta, p_delta = spearmanr(sub_m["mean_wup"], sub_m["delta"])
                rho_gain_delta, p_gain_delta = spearmanr(sub_m["delta"], sub_m["mean_gain"])
                correlations.append({
                    "classifier": cl,
                    "metric": m_name,
                    "rho_sim_vs_gain": rho_sample,
                    "p_sim_vs_gain": p_sample,
                    "rho_sim_vs_delta_gap": rho_delta,
                    "p_sim_vs_delta_gap": p_delta,
                    "rho_delta_gap_vs_gain": rho_gain_delta,
                    "p_delta_gap_vs_gain": p_gain_delta
                })
        else:
            correlations.append({
                "classifier": cl,
                "metric": "N/A",
                "rho_sim_vs_gain": rho_sample,
                "p_sim_vs_gain": p_sample,
                "rho_sim_vs_delta_gap": None,
                "p_sim_vs_delta_gap": None,
                "rho_delta_gap_vs_gain": None,
                "p_delta_gap_vs_gain": None
            })

    corr_df = pd.DataFrame(correlations).drop_duplicates()
    print(corr_df.to_string(index=False))

    # ---------------------------------------------------------
    # Analysis 3: Failure Analysis (Top 50 vs. Bottom 50 Classes)
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYSIS 3: Failure Analysis (Top 50 vs. Bottom 50 Classes by Gain)")
    print("="*80)

    failure_records = []
    for cl, group in full_df.groupby("classifier"):
        class_stats = group.groupby("class_id").agg(
            same_class_pct=("is_same_class", lambda x: np.mean(x) * 100.0),
            mean_wup_sim=("wup_sim", "mean"),
            mean_gain=("acc_gain", "mean")
        ).reset_index()

        # Attach class-level delta gap if available
        if not domain_gap_df.empty:
            cl_gaps = domain_gap_df[(domain_gap_df["classifier"] == cl) & (domain_gap_df["metric"] == "mmd")]
            class_stats = pd.merge(class_stats, cl_gaps[["class_id", "delta"]], on="class_id", how="left")
        else:
            class_stats["delta"] = np.nan

        sorted_classes = class_stats.sort_values(by="mean_gain", ascending=False)
        top_50 = sorted_classes.head(50)
        bottom_50 = sorted_classes.tail(50)

        failure_records.append({
            "classifier": cl,
            "group": "Top 50 Gain Classes",
            "same_class_pct": top_50["same_class_pct"].mean(),
            "mean_wup_sim": top_50["mean_wup_sim"].mean(),
            "mean_delta_gap": top_50["delta"].mean(),
            "mean_gain": top_50["mean_gain"].mean()
        })

        failure_records.append({
            "classifier": cl,
            "group": "Bottom 50 Gain Classes",
            "same_class_pct": bottom_50["same_class_pct"].mean(),
            "mean_wup_sim": bottom_50["mean_wup_sim"].mean(),
            "mean_delta_gap": bottom_50["delta"].mean(),
            "mean_gain": bottom_50["mean_gain"].mean()
        })

    fail_df = pd.DataFrame(failure_records)
    print(fail_df.to_string(index=False))

if __name__ == "__main__":
    main()

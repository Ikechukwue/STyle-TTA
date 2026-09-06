import json
import torch
from pathlib import Path
from tqdm import tqdm
from experiments.data import create_dataset
from torch.utils.data import DataLoader
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from torchvision.transforms import v2
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from config.helpers import load_json, get_y_true, get_top_k
from config.constants import ALL_CLASSIFIERS
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def export_retrieval_mapping(
    retriever, 
    content_dataloader, 
    output_path: str | Path = "./results/retrieval_mapping/retrieval_mapping.json",
    k: int = 64, 
    include_class: bool = False
):
    """
    Maps content image indices to a list of retrieved style image indices (or tuples).
    """
    mapping = {}
    
    print(f"Retrieving top-{k} style images for each content image...")
    
    # Assuming your dataloader returns (image, label, index) or similar. 
    # If it just returns (image, label), we can use enumerate(dataloader) as the sample_idx.
    device = getattr(retriever, "device", "cuda" if torch.cuda.is_available() else "cpu")

    for sample_idx, (image, _) in enumerate(tqdm(content_dataloader)):
        
        if image.dim() == 3:
            query = image.unsqueeze(0).to(device)
        else:
            query = image.to(device)
            
        # Retrieve the top-k style images
        indices, scores = retriever.retrieve(query, k=k)
        
        if include_class:
            # Map the retrieved index to its class label using retriever.labels
            # Convert to int() so it is JSON serializable
            retrieved_items = [
                (int(idx), int(retriever.labels[idx].item())) for idx in indices
            ]
        else:
            # Just store the IDs
            retrieved_items = [int(idx) for idx in indices]
            
        mapping[sample_idx] = retrieved_items
        
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Saved retrieval mapping to {output_path}")
    return mapping

def create_retieval_json(ds, split, method:str = "dino",
                         seed:int = 71397589, 
                         output_path:str = "results"):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=224),
    ])
    dataset = create_dataset(ds, "./data", split, transform)
    
    # Removed the nested dataloader, generator, and worker_init_fn
    data = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )

    ref_db = build_reference_db(
        dataset=ds,
        data_path="./data",
        input_size=224,
        seed=seed,
        split=split
    )
    
    print("Fetching Retriever...")
    retriever = build_retriever(
        strategy=method,
        db=ref_db,
        seed=seed,
        metric_type="ssim",
        embedding_model="vit_base_patch16_dinov3.lvd1689m",
        embedding_dir="./data/embeddings",
        dataset=ds,
        device="cuda",
        eval_split=split
    )
    
    export_retrieval_mapping(
        retriever=retriever,
        content_dataloader=data, 
        output_path=output_path,
        include_class=True
    )

#################################

from collections import Counter
import numpy as np

def retrieval_stats(retrieval_json):
    # Load retrieval mapping and ground truth labels
    data = load_json(retrieval_json)
    y_true = get_y_true("imagenet", "test_r")

    total_queries = len(data)
    if total_queries == 0:
        print("Retrieval JSON is empty.")
        return

    # Tracking metrics
    top1_class_matches = 0
    total_class_matches_in_k = 0
    total_elements_checked = 0
    
    style_id_counts = Counter()
    style_class_counts = Counter()
    per_query_match_percentages = []

    # JSON keys are always strings, so we loop over items
    for content_idx_str, retrieved_items in data.items():
        content_idx = int(content_idx_str)
        true_class = y_true[content_idx]
        
        if not retrieved_items:
            continue

        # 1. Top-1 Semantic Accuracy
        top1_id, top1_class = retrieved_items[0]
        if top1_class == true_class:
            top1_class_matches += 1

        # 2. Gather Top-K statistics
        k = len(retrieved_items)
        query_matches = 0
        
        for style_id, style_class in retrieved_items:
            style_id_counts[style_id] += 1
            style_class_counts[style_class] += 1
            
            if style_class == true_class:
                query_matches += 1
                total_class_matches_in_k += 1
        
        total_elements_checked += k
        per_query_match_percentages.append((query_matches / k) * 100)

    # Calculate overall metrics
    top1_match_rate = (top1_class_matches / total_queries) * 100
    global_precision_at_k = (total_class_matches_in_k / total_elements_checked) * 100
    avg_per_query_precision = np.mean(per_query_match_percentages)
    unique_styles_used = len(style_id_counts)
    unique_classes_used = len(style_class_counts)

    # Print the Analysis Report
    print("=" * 60)
    print("          RETRIEVAL STRATEGY DIAGNOSTICS          ")
    print("=" * 60)
    print(f"Total Content Queries Processed : {total_queries}")
    print(f"Top-1 Semantic Class Match Rate : {top1_match_rate:.2f}%")
    print(f"Global Precision@K Overlap      : {global_precision_at_k:.2f}%")
    print(f"Avg Class Match % Per Query    : {avg_per_query_precision:.2f}%")
    print("-" * 60)
    print(f"Unique Reference Images Picked  : {unique_styles_used}")
    print(f"Unique Reference Classes Picked : {unique_classes_used}")
    print("-" * 60)

    # Detect Hubness (retrieval bottlenecks)
    print("\n[Hub Detection] Top 5 Most Frequently Picked Reference Images:")
    for style_id, count in style_id_counts.most_common(5):
        percentage = (count / total_queries) * 100
        print(f"  • Style ID {style_id:5d} -> Picked {count:4d} times ({percentage:.1f}% of all queries)")

    print("\n[Bias Detection] Top 5 Most Frequently Picked Reference Classes:")
    for sys_class, count in style_class_counts.most_common(5):
        print(f"  • Class {sys_class:4d}    -> Selected {count:4d} times total across all neighbors")
    print("=" * 60)

    # Return metrics dictionary in case you want to plot them later
    return {
        "top1_match_rate": top1_match_rate,
        "global_precision_at_k": global_precision_at_k,
        "unique_styles_used": unique_styles_used,
        "style_id_counts": dict(style_id_counts)
    }

def analyze_neighborhood_dynamics(retrieval_json, y_true, k_list=[2, 4, 8, 16, 32, 64]):
    data = load_json(retrieval_json)
    total_queries = len(data)
    
    # ------------------------------------------------------------------
    # PART 1: Macro Neighborhood Scaling Metrics (2 to 64 views)
    # ------------------------------------------------------------------
    macro_stats = []
    
    for i in k_list:
        k = i-1
        presence_count = 0
        unique_classes_per_query = []
        true_class_proportions = []
        
        for content_idx_str, retrieved_items in data.items():
            content_idx = int(content_idx_str)
            true_class = y_true[content_idx]
            
            # Slice the neighborhood to current view size k
            neighborhood = retrieved_items[:k]
            if not neighborhood:
                continue
                
            neighbor_classes = [item[1] for item in neighborhood]
            
            # 1. Presence: Is the correct class in here at all?
            if true_class in neighbor_classes:
                presence_count += 1
                
            # 2. Diversity: How many unique classes are pulled in?
            unique_classes_per_query.append(len(set(neighbor_classes)))
            
            # 3. Dominance: What % of the neighborhood is the right class?
            true_count = neighbor_classes.count(true_class)
            true_class_proportions.append((true_count / k) * 100)
            
        macro_stats.append([
            f"{k} Views",
            f"{(presence_count / total_queries) * 100:.2f}%",
            f"{np.mean(unique_classes_per_query):.2f}",
            f"{np.mean(true_class_proportions):.2f}%"
        ])
        
    print("\n" + "="*70)
    print("             PART 1: NEIGHBORHOOD SCALING DYNAMICS")
    print("="*70)
    print(tabulate(macro_stats, headers=["Scale", "True Class Presence", "Avg Unique Classes", "True Class Dominance %"], tablefmt="grid"))

    # ------------------------------------------------------------------
    # PART 2: Class Contender & Rivalry Analysis (Evaluated at max K=64)
    # ------------------------------------------------------------------
    # Group all neighbors by the query's true class
    class_neighborhoods = {}
    for content_idx_str, retrieved_items in data.items():
        content_idx = int(content_idx_str)
        true_class = int(y_true[content_idx])
        
        if true_class not in class_neighborhoods:
            class_neighborhoods[true_class] = []
        
        # Pull classes from all retrieved style neighbors
        class_neighborhoods[true_class].extend([item[1] for item in retrieved_items])
        
    rivalry_report = []
    all_margins = []
    
    for target_class, all_neighbors in class_neighborhoods.items():
        if not all_neighbors:
            continue
            
        counts = Counter(all_neighbors)
        true_class_count = counts[target_class]
        
        # Find the top contender (the most frequent WRONG class)
        wrong_class_counts = {c: amt for c, amt in counts.items() if c != target_class}
        
        if wrong_class_counts:
            rival_class, rival_count = Counter(wrong_class_counts).most_common(1)[0]
        else:
            rival_class, rival_count = "None", 0
            
        # Margin of Safety: Positive means True Class dominates; Negative means Rival out-votes True Class
        margin = true_class_count - rival_count
        all_margins.append(margin)
        
        rivalry_report.append({
            "class_id": target_class,
            "true_count": true_class_count,
            "rival_id": rival_class,
            "rival_count": rival_count,
            "margin": margin
        })
        
    # Sort classes by who has the fiercest competition (lowest margin) 
    # and highest safety (highest margin)
    rivalry_report.sort(key=lambda x: x["margin"])
    
    print("\n" + "="*70)
    print("             PART 2: CLASS CONTENDER & RIVALRY ANALYSIS")
    print("="*70)
    print(f"Average Class Margin Overall: {np.mean(all_margins):.1f} votes")
    
    print("\nTOP 5 FIERCEST CLASS BATTLES (Lowest Margin / Heavy Contention):")
    fierce_table = []
    for item in rivalry_report[:5]:
        fierce_table.append([
            f"Class {item['class_id']}", item['true_count'], 
            f"Class {item['rival_id']}", item['rival_count'], item['margin']
        ])
    print(tabulate(fierce_table, headers=["True Class", "True Votes", "Top Rival Class", "Rival Votes", "Margin (True-Rival)"], tablefmt="simple"))

    print("\nTOP 5 SAFEST CLASSES (Highest Margin / Complete Dominance):")
    safe_table = []
    for item in rivalry_report[-5:][::-1]:
        safe_table.append([
            f"Class {item['class_id']}", item['true_count'], 
            f"Class {item['rival_id']}", item['rival_count'], item['margin']
        ])
    print(tabulate(safe_table, headers=["True Class", "True Votes", "Top Rival Class", "Rival Votes", "Margin (True-Rival)"], tablefmt="simple"))
    print("="*70 + "\n")


from scipy.stats import pointbiserialr

def compute_sample_correlations(
    retrieval_json, y_true, base_preds_json, tta_preds_json
):
    retrieval_data = load_json(retrieval_json)
    base_data = load_json(base_preds_json)
    tta_data = load_json(tta_preds_json)

    proportions = []
    has_majority = []
    base_corrects = []
    tta_corrects = []

    for content_idx_str, retrieved_items in retrieval_data.items():
        content_idx = int(content_idx_str)
        true_class = int(y_true[content_idx])

        if not retrieved_items:
            continue

        neighbor_classes = [item[1] for item in retrieved_items]
        k = len(retrieved_items)

        true_count = neighbor_classes.count(true_class)
        prop = true_count / k if k > 0 else 0.0
        
        # Strict majority (>50%)
        is_maj = 1 if prop > 0.5 else 0

        base_logits = base_data["predictions"][content_idx]["y_pred"]
        tta_logits = tta_data["predictions"][content_idx]["y_pred"]

        base_correct = 1 if int(np.argmax(base_logits)) == true_class else 0
        tta_correct = 1 if int(np.argmax(tta_logits)) == true_class else 0

        proportions.append(prop)
        has_majority.append(is_maj)
        base_corrects.append(base_correct)
        tta_corrects.append(tta_correct)

    has_majority = np.array(has_majority)
    proportions = np.array(proportions)
    base_corrects = np.array(base_corrects)
    tta_corrects = np.array(tta_corrects)

    def safe_pointbiserial(x, y):
        # Return 0 correlation if standard deviation of x or y is zero
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 1.0
        return pointbiserialr(x, y)

    # Correlation using continuous proportion (works even for random retrieval)
    r_prop_tta, p_prop_tta = safe_pointbiserial(proportions, tta_corrects)
    
    # Strict majority correlation (will safely return 0.0 for random instead of NaN)
    r_maj_tta, p_maj_tta = safe_pointbiserial(has_majority, tta_corrects)

    wrong_mask = base_corrects == 0
    if np.sum(wrong_mask) > 0:
        r_recovery, p_recovery = safe_pointbiserial(
            has_majority[wrong_mask], tta_corrects[wrong_mask]
        )
        r_prop_recovery, p_prop_recovery = safe_pointbiserial(
            proportions[wrong_mask], tta_corrects[wrong_mask]
        )
    else:
        r_recovery, p_recovery = 0.0, 1.0
        r_prop_recovery, p_prop_recovery = 0.0, 1.0

    return {
        "corr_majority_vs_tta_acc": r_maj_tta,
        "p_val_majority": p_maj_tta,
        "corr_proportion_vs_tta_acc": r_prop_tta,
        "corr_majority_vs_recovery": r_recovery,
        "p_val_recovery": p_recovery,
        "corr_proportion_vs_recovery": r_prop_recovery
    }
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def print_correlation_summary(all_correlations):
    """
    Prints a formatted table summarizing overall TTA accuracy correlation
    and baseline recovery correlation across classifiers and strategies.
    """
    print("\n" + "=" * 85)
    print("      SAMPLE-LEVEL RETRIEVAL MAJORITY vs. ACCURACY CORRELATION ANALYSIS")
    print("=" * 85)

    headers = [
        "Classifier", 
        "Strategy", 
        "r(Majority, TTA Acc)", 
        "p-val", 
        "r(Majority, Recovery)", 
        "p-val"
    ]
    
    table_data = []
    for (cl, method), res in all_correlations.items():
        table_data.append([
            cl,
            method,
            f"{res['corr_majority_vs_tta_acc']:.4f}",
            f"{res['p_val_majority']:.2e}",
            f"{res['corr_majority_vs_recovery']:.4f}",
            f"{res['p_val_recovery']:.2e}"
        ])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("=" * 85 + "\n")


def plot_correlation_comparison(all_correlations, output_path="./results/retrieval_mapping"):
    """
    Plots a bar chart comparing the recovery correlation across strategies.
    """
    classifiers = list(dict.fromkeys([k[0] for k in all_correlations.keys()]))
    strategies = list(dict.fromkeys([k[1] for k in all_correlations.keys()]))

    x = np.arange(len(classifiers))
    width = 0.25

    plt.figure(figsize=(12, 6))

    for i, strategy in enumerate(strategies):
        recovery_corrs = [
            all_correlations.get((cl, strategy), {}).get("corr_majority_vs_recovery", 0.0)
            for cl in classifiers
        ]
        plt.bar(x + i * width, recovery_corrs, width, label=f"Strategy: {strategy}")

    plt.xlabel("Classifier Backbones", fontweight="bold")
    plt.ylabel("Point-Biserial Correlation (r)", fontweight="bold")
    plt.title("Correlation Between True-Class Majority Retrieval and Baseline Failure Recovery")
    plt.xticks(x + width, classifiers, rotation=15)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    out_file = f"{output_path}/majority_recovery_correlation.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved correlation plot to {out_file}")




def plot_class_dominance_ranking(
    json_paths, y_true, output_path="./results/retrieval_mapping"
):
    strategies_data = {}
    output_name = f"{output_path}/dominance_ranking.png"
    for name, path in json_paths.items():
        data = load_json(path)
        class_neighborhoods = {}
        for content_idx_str, retrieved_items in data.items():
            content_idx = int(content_idx_str)
            target_class = int(y_true[content_idx])

            if target_class not in class_neighborhoods:
                class_neighborhoods[target_class] = []
            class_neighborhoods[target_class].extend(
                [item[1] for item in retrieved_items]
            )

        dominance_values = []
        for target_class, all_neighbors in class_neighborhoods.items():
            if not all_neighbors:
                continue
            true_count = all_neighbors.count(target_class)
            total_votes = len(all_neighbors)
            dominance = true_count / total_votes if total_votes > 0 else 0
            dominance_values.append(dominance)

        dominance_values.sort(reverse=True)
        strategies_data[name] = dominance_values

    plt.figure(figsize=(10, 6))

    for name, dominance_values in strategies_data.items():
        x = np.arange(len(dominance_values))
        plt.plot(x, dominance_values, label=name, linewidth=2)

    plt.xlabel("Sorted Classes")
    plt.ylabel("Class Dominance")
    plt.title("Class Dominance Ranking Across Retrieval Strategies")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

if __name__ == "__main__":
    json_paths = {}
    all_correlations = {}

    for method in ["dino", "random", "balanced_random"]:
        print(f"---- Analysis for {method} ----")
        path = f"/home/stud/nemmler/retristyle/results/retrieval_mapping/imagenet/retrieval_mapping_{method}_test_r_s71397589.json"
        y_true = get_y_true("imagenet", "test_r")
        
        analyze_neighborhood_dynamics(path, y_true)
        json_paths[method] = path

        for cl in ALL_CLASSIFIERS:
            base_path = f"./results/geometric_tta/tta_inference/predictions/imagenet/test_r/{cl}_geometric_zero_nviews1_seed71397589.json"
            ood_path = f"./results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/{cl}_adain_tta_zero_{method}_nrefs32_seed71397589.json"
            
            res = compute_sample_correlations(path, y_true, base_path, ood_path)
            all_correlations[(cl, method)] = res

    # Print organized report
    print_correlation_summary(all_correlations)

    # Plot graphs
    plot_class_dominance_ranking(json_paths, y_true)
    plot_correlation_comparison(all_correlations)

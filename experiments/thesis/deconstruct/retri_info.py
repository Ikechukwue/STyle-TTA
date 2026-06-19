import json
import torch
from pathlib import Path
from tqdm import tqdm
from experiments.data import create_dataset
from torch.utils.data import DataLoader
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from torchvision.transforms import v2
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from config.helpers import load_json, get_y_true
from tabulate import tabulate

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

def create_retieval_json(method:str = "dino",
                         seed:int = 71397589, 
                         output_path:str = "results"):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=224),
    ])
    dataset = create_dataset("imagenet", "./data", "test_r", transform)
    
    # Removed the nested dataloader, generator, and worker_init_fn
    data = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )

    ref_db = build_reference_db(
        dataset="imagenet",
        data_path="./data",
        input_size=224,
        seed=seed,
        split="test_r"
    )
    
    print("Fetching Retriever...")
    retriever = build_retriever(
        strategy=method,
        db=ref_db,
        seed=seed,
        metric_type="ssim",
        embedding_model="vit_base_patch16_dinov3.lvd1689m",
        embedding_dir="./data/embeddings",
        dataset="imagenet",
        device="cuda",
        eval_split="test_r"
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

if __name__ == "__main__":
    for method in ["dino", "random", "balanced_random"]:
        print(f"----Analysis for {method}----")
        path = f"/home/stud/nemmler/retristyle/results/retrieval_mapping/retrieval_mapping_{method}_test_r_s71397589.json"
        y_true = get_y_true("imagenet", "test_r")
        analyze_neighborhood_dynamics(path,y_true)

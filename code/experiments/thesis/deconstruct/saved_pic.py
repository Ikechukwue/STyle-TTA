from pathlib import Path
from code.config.helpers import load_json
import numpy as np 

from collections import Counter


def print_style_influence(base_pred_path: Path, tta_pred_path: Path, mapping_json_path: Path, n_refs: int):
    mapping = load_json(mapping_json_path)
    base_data = load_json(base_pred_path)["predictions"]
    tta_data = load_json(tta_pred_path)["predictions"]

    base_lookup = {item["sample_idx"]: (item["y_true"], np.argmax(item["y_pred"])) for item in base_data}
    tta_lookup = {item["sample_idx"]: np.argmax(item["y_pred"]) for item in tta_data}

    saved_by_matching_style = 0
    saved_and_dominant = 0
    shifted_to_predominant_style = 0
    total_saved = 0
    total_shifts = 0

    for idx_str, style_info in mapping.items():
        idx = int(idx_str)
        if idx not in base_lookup or idx not in tta_lookup:
            continue
            
        gt, b_pred = base_lookup[idx]
        t_pred = tta_lookup[idx]
        
        active_refs = style_info[:n_refs]
        if not active_refs:
            continue
            
        ref_classes = [ref[1] for ref in active_refs]
        
        class_counts = Counter(ref_classes)
        predominant_style_cls, _ = class_counts.most_common(1)[0]

        # Saved: Baseline incorrect -> TTA correct
        if b_pred != gt and t_pred == gt:
            total_saved += 1
            if gt in ref_classes:
                saved_by_matching_style += 1
                if predominant_style_cls == gt:
                    saved_and_dominant += 1

        # Shifted: Prediction changed between baseline and TTA
        if b_pred != t_pred:
            total_shifts += 1
            if t_pred == predominant_style_cls:
                shifted_to_predominant_style += 1

    pct_saved = (saved_by_matching_style / total_saved * 100) if total_saved > 0 else 0
    pct_dominant = (saved_and_dominant / saved_by_matching_style * 100) if saved_by_matching_style > 0 else 0
    pct_shifted = (shifted_to_predominant_style / total_shifts * 100) if total_shifts > 0 else 0

    print(f"--- Style Influence Breakdown (n_refs={n_refs}) ---")
    print(f"Saved Correctly: {saved_by_matching_style}/{total_saved} ({pct_saved:.1f}%) were fixed when at least one style image matched the TRUE class.")
    print(f"  -> From those saved with at least one correct class style image, {saved_and_dominant}/{saved_by_matching_style} ({pct_dominant:.1f}%) had the correct class as the dominant class.")
    print(f"Class Pull Effect: {shifted_to_predominant_style}/{total_shifts} ({pct_shifted:.1f}%) of prediction changes moved toward the PREDOMINANT style class.")

if __name__=="__main__":
    base_pred = "/home/stud/nemmler/style_tta/results/baseline/tta_inference/predictions/imagenet/test_r/resnet18_geometric_vanilla_nviews1_seed71397589.json"
    for ref in [1, 3, 7, 15]:
        tta_path = f"/home/stud/nemmler/style_tta/results/ablation/style_tta/tta_inference/predictions/imagenet/test_r/resnet18_style_tta_vanilla_dino_nrefs{ref+1}_seed71397589.json"
        mapping_path = "/home/stud/nemmler/style_tta/results/retrieval_mapping/retrieval_mapping_dino_test_r_s71397589.json"
        print_style_influence(base_pred, tta_path, mapping_path, ref)

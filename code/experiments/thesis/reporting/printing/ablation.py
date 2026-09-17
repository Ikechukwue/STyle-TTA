import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from code.config.constants import ALL_CLASSIFIERS
from code.config.helpers import load_json, get_baseline_results, calc_top_k
# Helper: Map raw filenames to readable labels
def get_classifier_name(cls: str) -> tuple[str, str]:
    mapping = {
        "resnet18": ("ResNet-18", "CNN"),
        "densenet121": ("DenseNet-121", "CNN"),
        "vit_base_patch16_224": ("ViT-B/16 (224)", "Vision Transformer"),
        "swin_base_patch4_window7_224": ("Swin-B (224)", "Vision Transformer"),
        "ViT-B-16": ("CLIP ViT-B/16", "Vision-Language Model"),
        "ViT-B-16@Zero": ("CLIP ViT-B/16 (Zero-Shot)", "Vision-Language Model"),
        "dinov2_vitb14": ("DINOv2 ViT-B/14", "Foundation Model"),
        "vit_base_patch16_dinov3_lvd1689m": ("DINOv3 ViT-B/16", "Foundation Model"),
    }
    return mapping.get(cls, (cls, "Unknown"))

def print_all_nrefs_metrics(results_dir: Path, dataset, split):
    """
    Prints a terminal table of Top-1 accuracy for all classifiers 
    across adain, retristyle, and geometric methods.
    """
    methods = {
        "ablation/adain_tta": "{cl}_adain_tta_zero_dino_nrefs{rfs}_seed71397589.json",
        "ablation/retristyle": "{cl}_retristyle_zero_dino_nrefs{rfs}_seed71397589.json",
        "geometric_tta": "{cl}_geometric_zero_nviews{rfs}_seed71397589.json"
    }
    
    axis_values = [2, 4, 8, 16, 32, 64]

    for method, template in methods.items():
        print(f"\n{'='*15} Method: {method} {'='*15}")
        header = f"{'Classifier':<20} | " + " | ".join([f"T1@{r:<2}" for r in axis_values])
        print(header)
        print("-" * len(header))

        for cl in ALL_CLASSIFIERS:
            vals = []
            for rfs in axis_values:

                if method == "ablation/retristyle" and rfs > 16:
                    continue
                # Build path
                f = results_dir / method / f"tta_inference/results/{dataset}/{split}" / template.format(cl=cl, rfs=rfs)
                
                if not f.exists():
                    vals.append(None)
                    continue
                
                try:
                    data = load_json(f)
                    acc = data.get("metrics", {}).get("accuracy")
                    vals.append(acc * 100 if acc is not None else None)
                except Exception:
                    vals.append(None)

            # Print formatted row
            row = [f"{get_classifier_name(cl)[0]:<20}"]
            row += [f"{v:6.1f}" if v is not None else "     -" for v in vals]
            print(" | ".join(row))
            
def print_all_topk_metrics(results_dir: Path, dataset, split):
    methods = {
        "ablation/adain_tta": "{cl}_adain_tta_zero_dino_nrefs64_seed71397589.json",
        "ablation/retristyle": "{cl}_retristyle_zero_dino_nrefs16_seed71397589.json",
        "geometric_tta": "{cl}_geometric_zero_nviews64_seed71397589.json"
    }

    for method, template in methods.items():
        print(f"\n{'='*10} {method} (Top-1/Top-5) {'='*10}")
        header = f"{'Classifier':<20} | T1 (Avg) | T5 (Avg) | Base1 | Base5"
        print(header)
        print("-" * 50)

        for cl in ALL_CLASSIFIERS:
            t1_list, t5_list = [], []
            f_name = template.format(cl=cl)
            f = results_dir / method / f"tta_inference/results/{dataset}/{split}" / f_name
            if not f.exists(): continue
            
            # Top-1
            data = load_json(f)
            t1 = data.get("metrics", {}).get("accuracy")
            if t1 is not None: t1_list.append(t1 * 100)
            
            # Top-5
            preds_p = results_dir / method / f"tta_inference/predictions/{dataset}/{split}" / f_name
            t5 = calc_top_k(preds_p, 5)
            if t5 is not None: t5_list.append(t5)

            # Get Baseline Top-5
            base_p = get_baseline_results(template.format(cl=cl), False, split)
            base_val = load_json(base_p).get("metrics", {}).get("accuracy") * 100 if base_p.exists() else 0.0

            base_p5 = get_baseline_results(template.format(cl=cl), True, split)
            base_val5 = calc_top_k(base_p5, 5)

            # Compute row averages
            avg_t1 = np.mean(t1_list) if t1_list else 0.0
            avg_t5 = np.mean(t5_list) if t5_list else 0.0
            
            print(f"{get_classifier_name(cl)[0]:<20} | {avg_t1:6.1f} | {avg_t5:6.1f} | {base_val:6.1f} | {base_val5:6.1f}")

# Entry point function to trigger the output
def generate_ablation_report():
    results_dir = Path("./results")
    dataset= "imagenet"
    split = "test_r"
    print("Generating Ablation Summary Report...")
    print_all_nrefs_metrics(results_dir,dataset, split)
    print_all_topk_metrics(results_dir, dataset, split)
if __name__ == "__main__":
    generate_ablation_report()

import json
import math

def load_data(json_file_path: str) -> dict:
    with open(json_file_path, "r") as f:
        return json.load(f)

def format_value(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"

def format_delta(delta: float, decimals: int = 3) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{decimals}f}"

def generate_latex_table(data: dict) -> str:
    backbones = [
        ("resnet18", "ResNet-18"),
        ("densenet121", "DenseNet-121"),
        ("vit_base_patch16_224", "ViT-B/16 (Supervised)"),
        ("swin_base_patch4_window7_224", "Swin-B"),
        ("ViT-B-16", "CLIP (ViT-B/16)"),
        ("dinov2_vitb14", "DINOv2 (ViT-B/14)"),
    ]

    metrics = ["mmd", "wasserstein", "kl_symmetric"]
    ood_data = data.get("ood", {})
    base_data = data.get("baseline", {})

    max_deltas = {}
    min_deltas = {}
    for metric in metrics:
        metric_key = f"{metric}_mean"
        deltas = [
            ood_data.get(key, {}).get(metric_key, 0.0) - base_data.get(key, {}).get(metric_key, 0.0)
            for key, _ in backbones
        ]
        max_deltas[metric] = max(deltas)
        min_deltas[metric] = min(deltas)
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Global feature-space domain discrepancy metrics across evaluated backbones on ImageNet-1K vs. ImageNet-R over 200 classes. The net shift ($\Delta$) isolates distribution shift from in-distribution sampling noise.}")
    latex.append(r"\label{tab:domain_gap_summary}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l ccc ccc ccc}")
    latex.append(r"\toprule")
    latex.append(r" & \multicolumn{3}{c}{\textbf{MMD}} & \multicolumn{3}{c}{\textbf{Wasserstein ($W_2$)}} & \multicolumn{3}{c}{\textbf{Symmetric KL}} \\")
    latex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    latex.append(r"\textbf{Backbone} & \textbf{OOD} & \textbf{Base} & $\mathbf{\Delta}$ & \textbf{OOD} & \textbf{Base} & $\mathbf{\Delta}$ & \textbf{OOD} & \textbf{Base} & $\mathbf{\Delta}$ \\")
    latex.append(r"\midrule")

    for key, display_name in backbones:
        row_cells = [display_name]
        
        for metric in metrics:
            metric_key = f"{metric}_mean"
            ood_val = ood_data.get(key, {}).get(metric_key, 0.0)
            base_val = base_data.get(key, {}).get(metric_key, 0.0)
            delta_val = ood_val - base_val
            
            formatted_delta = format_delta(delta_val)
            if math.isclose(delta_val, max_deltas[metric], rel_tol=1e-5):
                formatted_delta = f"\\colorbox{{lightgray}}{{{formatted_delta}}}"

            elif math.isclose(delta_val, min_deltas[metric], rel_tol=1e-5):
                formatted_delta = f"\\fcolorbox{{lightgray}}{{white}}{{{formatted_delta}}}"
            row_cells.extend([
                format_value(ood_val),
                format_value(base_val),
                formatted_delta
            ])
        
        latex.append(" & ".join(row_cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)

if __name__ == "__main__":
    data = load_data("/home/stud/nemmler/retristyle/results/domain_gap/feature_space/test_r/full_cross_domain_gap.json")
    latex_code = generate_latex_table(data)
    print(latex_code)

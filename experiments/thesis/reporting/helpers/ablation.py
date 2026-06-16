from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import re
from config.helpers import load_json, get_top_k, top_k_acc, calc_top_k, get_baseline_results
from config.constants import TTA_STRATEGIES, ALL_CLASSIFIERS
import pandas as pd
def plot_ablation_bars(results_dir: Path, output_dir: Path, ablation_type: str, method: str, args, 
                      best_retrieval="dino", best_eval="zero", best_n_refs=16):
    """
    Generates a bar chart isolating an ablation axis and draws a baseline shadow layer.
    """
    res_suffix = method
    
    results_base = results_dir / res_suffix
    if (results_base / "tta_inference" / "results").exists():
        metrics_dir = results_base / "tta_inference" / "results" / args.dataset / args.split
        preds_dir = results_base / "tta_inference" / "predictions" / args.dataset / args.split
    else:
        metrics_dir = results_base
        preds_dir = results_base 

    res_suffix = res_suffix.replace("/", "_")
    files = list(metrics_dir.glob("**/*.json"))
    if not files:
        print(f"  [skip] No ablation results found in {metrics_dir} for {ablation_type}")
        return

    grouped: Dict[str, Dict[str, List[float]]] = {}
    # Structure: baseline_storage[classifier][eval_strategy] = list of float accuracies
    baseline_storage: Dict[str, Dict[str, List[float]]] = {}
    for k in [1, 5]:
        for f in files:
            try:
                data = load_json(f)
            except Exception:
                continue
                        
            clf = data.get("classifier", "?")
            retr = data.get("retrieval_strategy", "none")
            ev_strat = data.get("eval_strategy", "none")
            nr = data.get("n_refs", 0)
            
            # --- Strict Filtering Axes ---
            if ablation_type == "retrieval":
                if str(ev_strat) != str(best_eval) or int(nr) != int(best_n_refs): continue
                key = retr
            elif ablation_type == "eval":
                if str(retr) != str(best_retrieval) or int(nr) != int(best_n_refs): continue
                key = ev_strat
            elif ablation_type == "nrefs":
                if str(retr) != str(best_retrieval) or str(ev_strat) != str(best_eval): continue
                key = str(nr)
            else:
                continue

            # --- Calculate Target Accuracy Value ---
            if k == 1:
                acc = data.get("metrics", {}).get("accuracy")
                if acc is not None: acc = acc * 100
            else:
                pred_file_path = preds_dir / f.name
                acc = calc_top_k(pred_file_path, k)
                
            if acc is None: continue
            grouped.setdefault(clf, {}).setdefault(key, []).append(acc)

            # ---- Gather Baseline Data ----
            try:
                if k == 1:
                    base_metrics_path = get_baseline_results(f.name, False, args.split)
                    base_data = load_json(base_metrics_path)
                    base_acc = base_data.get("metrics", {}).get("accuracy")
                    if base_acc is not None: base_acc *= 100
                else:
                    base_pred_path = get_baseline_results(f.name, True, args.split)
                    base_acc = calc_top_k(base_pred_path, k)
                
                if base_acc is not None:
                    # Store baselines matching the current evaluation strategy setup
                    baseline_storage.setdefault(clf, {}).setdefault(ev_strat, []).append(base_acc)
            except Exception as e:
                # Baseline file might be missing for a specific seed/setting
                pass

        if not grouped:
            print(f"  [skip] No matching files survived filtering for axis: {ablation_type} (Top-{k})")
            return

        # --- Plot Generation Pipeline ---
        for clf, strategy_dict in grouped.items():
            labels = sorted(strategy_dict.keys())
            means = [np.mean(strategy_dict[k_val]) for k_val in labels]
            stds = [np.std(strategy_dict[k_val]) for k_val in labels]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            
            # 1. Plot the Primary Bars
            bars = ax.bar(labels, means, yerr=stds, capsize=4,
                        color=plt.cm.Set2(np.arange(len(labels)) % 8), width=0.5, zorder=3)
            
            # 2. INJECT BASELINE SHADOW LAYER
            # Determine the baseline reference value for this classifier
            current_eval_mode = best_eval if ablation_type != "eval" else None
            
            base_values = []
            if current_eval_mode:
                base_values = baseline_storage.get(clf, {}).get(current_eval_mode, [])
            else:
                # For the eval strategy axis, aggregate baselines across all visible strategies
                for ev_mode in labels:
                    base_values.extend(baseline_storage.get(clf, {}).get(ev_mode, []))

            print(f"DEBUG {clf}: Found {len(baseline_storage)} baseline values -> {base_values}")

            if base_values:
                mean_baseline = np.mean(base_values)
                
                # Draw a dashed line indicating the source baseline performance floor
                ax.axhline(mean_baseline, color="gray", linestyle="--", linewidth=1.5, 
                        label=f"Baseline ({mean_baseline:.1f}%)", zorder=2)
                
                # Draw a visual "shadow" band across the plot
                ax.axhspan(0, mean_baseline, color="gray", alpha=0.07, zorder=1)
                ax.legend(loc="upper left")

            # Formatting tweaks
            ax.set_ylabel(f"Top-{k} Accuracy (%)")
            ax.set_title(f"{clf} — {ablation_type.capitalize()} Axis (Top-{k} Accuracy)")
            
            # Dynamic Limits
            all_points = means + (base_values if base_values else [])
            ymin = max(0, min(all_points) - 5)
            ymax = min(100, max(all_points) + 5)
            ax.set_ylim(bottom=ymin, top=ymax)
            ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

            for bar, m in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight='bold', zorder=4)

            fig.tight_layout()
            clf_clean = clf.replace("/", "_").replace("-", "_")
            out_dir = output_dir / method / args.split
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out = out_dir / f"ablation_{ablation_type}_{clf_clean}_top{k}.pdf"
            fig.savefig(out, bbox_inches='tight')
            plt.close(fig)
            print(f"  [saved] Grouped Bar Plot with Baseline (Top-{k}): {out}")

def plot_nrefs_sweep(results_dir: Path, output_dir: Path, args, method:str,
                     best_retrieval="dino", best_eval="zero"):
    """
    Plots validation accuracies tracking alongside increasing scale parameters,
    filtering out mixed experimental configurations.
    """
    res_suffix = method 
    abl_dir = results_dir / res_suffix
    if (results_dir / res_suffix / "tta_inference" / "results").exists():
        abl_dir = results_dir / res_suffix / "tta_inference" / "results" / args.dataset / args.split
    res_suffix = res_suffix.replace("/", "_")
    files = list(abl_dir.glob("**/*.json"))
    if not files:
        print("  [skip] No n_refs sweep results found")
        return

    # Structure: grouped[(n_refs, classifier)] = [accuracies]
    grouped: Dict[Tuple[int, str], List[float]] = {}
    
    for f in files:
        data = load_json(f)
            
        retr = data.get("retrieval_strategy", "none")
        ev_strat = data.get("eval_strategy", "none")
        
        # CRITICAL FILTER: Only process configurations using the true isolated baseline components
        if str(retr) != str(best_retrieval) or str(ev_strat) != str(best_eval):
            continue
        if method == "geometric_tta":
            match = re.search(r"nviews(\d+)", f)
            nr = int(match.group(1))
        else:   
            nr = data.get("n_refs") 
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        
        if nr is not None and acc is not None:
            grouped.setdefault((int(nr), clf), []).append(acc * 100)

    if not grouped:
        print("  [skip] No pure n_refs configurations matching baseline evaluation parameters survived.")
        return

    classifiers = sorted(set(k[1] for k in grouped))
    nrefs_available = sorted(set(k[0] for k in grouped))

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for clf in classifiers:
        means = []
        stds = []
        valid_nrefs = []
        
        for nr in nrefs_available:
            scores = grouped.get((nr, clf))
            if scores:  # Verify this classifier hit this tracking resolution step
                means.append(np.mean(scores))
                stds.append(np.std(scores))
                valid_nrefs.append(nr)
                
        ax.errorbar(valid_nrefs, means, yerr=stds, marker="o", markersize=5, 
                    linewidth=1.5, label=clf, capsize=4)

    ax.set_xlabel("Number of Style References ($n_{refs}$)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(f"Style-Transfer TTA Scale Sweep (Using {best_retrieval.upper()} + {best_eval.capitalize()})", fontsize=11, fontweight='bold')
    
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(nrefs_available)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    out_dir = output_dir / method / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{res_suffix}_nrefs_sweep_filtered.pdf"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] Pure Line Plot: {out}")

def plot_topk_confidence(results_dir: Path, output_dir: Path, args, method: str, k: int = 5):
    """
    Reads JSON files, extracts top-K confidences using get_top_k, 
    and plots a line/bar chart showing average confidence drop-off per Rank.
    """
    res_suffix = method 
    abl_dir = results_dir / res_suffix

    if (results_dir / res_suffix / "tta_inference" / "predictions").exists():
        abl_dir = results_dir / res_suffix / "tta_inference" / "predictions" / args.dataset / args.split
    res_suffix = res_suffix.replace("/", "_")
    files = list(abl_dir.glob("**/*.json"))
    if not files:
        print(f"  [skip] No results found in {abl_dir} for Top-{k} visualization")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_generated = False

    for f in files:
        data = load_json(f)
            
        # Ensure the file contains the required raw prediction payload
        if "predictions" not in data or not data["predictions"]:
            continue
            
        clf = data["config"].get("classifier", "Unknown")
        retr = data["config"].get("retrieval_strategy", "none")
        ev_strat = data["config"].get("eval_strategy", "none")
        
        # Calculate top-k confidences
        _, final_confidences = get_top_k(data, k=k)
        
        # Calculate mean confidence and standard deviation for each rank position (1 to K)
        mean_confidences = np.mean(final_confidences, axis=0)
        std_confidences = np.std(final_confidences, axis=0)
        
        # Convert confidences to percentages if they are raw probabilities (0.0 - 1.0)
        if np.max(mean_confidences) <= 1.0:
            mean_confidences *= 100
            std_confidences *= 100

        ranks = np.arange(1, len(mean_confidences) + 1)
        label_text = f"{clf} ({retr}+{ev_strat})"
        
        # Plot confidence drop-off curve for this run
        ax.errorbar(ranks, mean_confidences, yerr=std_confidences, marker="o", 
                    linestyle="-", linewidth=1.5, capsize=3, label=label_text)
        plot_generated = True

    if not plot_generated:
        print(f"  [skip] No files with valid 'predictions' data found for Top-{k} plot.")
        return

    ax.set_xlabel("Prediction Rank (Top-K)")
    ax.set_ylabel("Mean Confidence (%)")
    ax.set_title(f"Top-{k} Prediction Confidence Drop-off ({method.capitalize()})", fontweight='bold')
    ax.set_xticks(ranks)
    ax.set_xlim(0.5, k + 0.5)
    ymin = max(0, min(mean_confidences) - 5)
    ymax = min(100, max(mean_confidences) + 5)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.3)
    #ax.legend(fontsize=8, loc="best")
    
    fig.tight_layout()
    out_dir = output_dir / "topk" / method / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{res_suffix}_top{k}_confidence.pdf"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] Top-{k} Confidence Plot: {out}")

def create_summary(results_dir: Path):
    rows = []

    methods = [
        "baseline",
        "geometric_tta",
        "adain_tta",
        "retristyle",
    ]
    ablation_methods = ["retristyle", "adain_tta"]
    for method in tqdm(methods, desc="Going through methods:"):
        
        method_name = Path("ablation/" + method) if method in ablation_methods else method 
        metrics_dir = (
            results_dir
            / method_name
            / "tta_inference"
            / "results"
            / "imagenet"
            / "test_r"
        )
        pred_dir = (
            results_dir
            / method_name
            / "tta_inference"
            / "predictions"
            / "imagenet"
            / "test_r"
        )

        for f in tqdm(metrics_dir.glob("*.json"), desc="Going through files:", leave=True):

            data = load_json(f)

            row = {
                "method": method,
                "classifier": data.get("classifier"),
                "top1": data.get("metrics", {}).get("accuracy"),
            }

            m = re.search(r"seed(\d+)", Path(f).name)
            row["seed"] = int(m.group(1)) if m else None

            if method == "geometric_tta":
                m = re.search(r"nviews(\d+)", Path(f).name)
                row["n_views"] = int(m.group(1)) if m else None
            else:
                row["retrieval"] = data.get("retrieval_strategy")
                row["eval"] = data.get("eval_strategy")
                row["n_refs"] = data.get("n_refs")

            top_k = calc_top_k(pred_dir / Path(f).name)
            row["top_5"] = top_k
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("master_sum.csv", index=False)

if __name__ == "__main__":
    res = Path("/home/stud/nemmler/retristyle/results")
    create_summary(res)

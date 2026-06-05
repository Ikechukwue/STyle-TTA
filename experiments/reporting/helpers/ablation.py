from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_ablation_bars(results_dir: Path, output_dir: Path, ablation_type: str, method:str, args, 
                       best_retrieval="dino", best_eval="zero", best_n_refs=16):
    """
    Generates a bar chart cleanly isolating a single ablation axis.
    Filters out background sweep noise and groups data clearly.
    """
    res_suffix = method if not method == 'geometric' else 'geometric_tta'
    abl_dir = results_dir / res_suffix

    if (results_dir / res_suffix / "tta_inference" / "results").exists():
        abl_dir = results_dir / res_suffix / "tta_inference" / "results" / args.dataset / args.split
        
    files = list(abl_dir.glob("**/*.json"))
    if not files:
        print(f"  [skip] No ablation results found in {abl_dir} for {ablation_type}")
        return

    # Structure: grouped[classifier][strategy_key] = [accuracies]
    grouped: Dict[str, Dict[str, List[float]]] = {}
    
    for f in files:
        try:
            import json
            with open(f, 'r') as fh:
                data = json.load(fh)
        except Exception:
            continue
            
        # Extract metrics safely
        acc = data.get("metrics", {}).get("accuracy")
        if acc is None:
            continue
            
        clf = data.get("classifier", "?")
        retr = data.get("retrieval_strategy", "none")
        ev_strat = data.get("eval_strategy", "none")
        nr = data.get("n_refs", 0)
        
        # --- Strict Filtering Axes ---
        if ablation_type == "retrieval":
            # Isolate retrieval: hold evaluation strategy and n_refs steady
            if str(ev_strat) != str(best_eval) or int(nr) != int(best_n_refs):
                continue
            key = retr
            
        elif ablation_type == "eval":
            # Isolate evaluation: hold retrieval strategy and n_refs steady
            if str(retr) != str(best_retrieval) or int(nr) != int(best_n_refs):
                continue
            key = ev_strat
            
        elif ablation_type == "nrefs":
            # Isolate n_refs: hold retrieval and evaluation strategies steady
            if str(retr) != str(best_retrieval) or str(ev_strat) != str(best_eval):
                continue
            key = str(nr)
        else:
            continue

        grouped.setdefault(clf, {}).setdefault(key, []).append(acc * 100)

    if not grouped:
        print(f"  [skip] No matching files survived the clean filtering for axis: {ablation_type}")
        return

    # Create a plot for each classifier so data remains unpolluted
    for clf, strategy_dict in grouped.items():
        labels = sorted(strategy_dict.keys())
        means = [np.mean(strategy_dict[k]) for k in labels]
        stds = [np.std(strategy_dict[k]) for k in labels]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, means, yerr=stds, capsize=4,
                      color=plt.cm.Set2(np.arange(len(labels)) % 8), width=0.5)
        
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{clf} — {ablation_type.capitalize()} Strategy Ablation")
        
        # Dynamic padding for the Y-axis label spacing
        ymin = max(0, min(means) - 5)
        ymax = min(100, max(means) + 5)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Draw values cleanly above individual bars
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight='bold')

        fig.tight_layout()
        clf_clean = clf.replace("/", "_").replace("-", "_")
        out_dir = output_dir / res_suffix / args.split
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"ablation_{ablation_type}_{clf_clean}.pdf"
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f"  [saved] Grouped Bar Plot: {out}")

def plot_nrefs_sweep(results_dir: Path, output_dir: Path, args, method:str,
                     best_retrieval="dino", best_eval="zero"):
    """
    Plots validation accuracies tracking alongside increasing scale parameters,
    filtering out mixed experimental configurations.
    """
    res_suffix = method if not method == 'geometric' else 'geometric_tta'
    abl_dir = results_dir / res_suffix
    if (results_dir / res_suffix / "tta_inference" / "results").exists():
        abl_dir = results_dir / res_suffix / "tta_inference" / "results" / args.dataset / args.split
        
    files = list(abl_dir.glob("**/*.json"))
    if not files:
        print("  [skip] No n_refs sweep results found")
        return

    # Structure: grouped[(n_refs, classifier)] = [accuracies]
    grouped: Dict[Tuple[int, str], List[float]] = {}
    
    for f in files:
        try:
            import json
            with open(f, 'r') as fh:
                data = json.load(fh)
        except Exception:
            continue
            
        retr = data.get("retrieval_strategy", "none")
        ev_strat = data.get("eval_strategy", "none")
        
        # CRITICAL FILTER: Only process configurations using the true isolated baseline components
        if str(retr) != str(best_retrieval) or str(ev_strat) != str(best_eval):
            continue
        
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
    out_dir = output_dir / "ablation" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{res_suffix}_nrefs_sweep_filtered.pdf"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] Pure Line Plot: {out}")


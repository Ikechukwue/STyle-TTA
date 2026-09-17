from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import re
from code.config.helpers import load_json, get_top_k, top_k_acc, calc_top_k, get_baseline_results, get_classifier_name
from code.config.constants import TTA_STRATEGIES, ALL_CLASSIFIERS, ALL_SEEDS, DEFAULT_SEED
from code.config.paths import OUTPUT_PATH
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

def plot_ablation_lines(results_dir: Path, output_dir: Path, ablation_type: str, method: str, args, 
                       best_retrieval="dino", best_eval="zero", best_n_refs=16):
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
        return

    for k in [1, 5]:
        grouped: Dict[str, Dict[str, List[float]]] = {}
        baseline_storage: Dict[str, Dict[str, List[float]]] = {}
        
        for f in files:
            try:
                data = load_json(f)
            except Exception:
                continue
                        
            clf = data.get("classifier", "?")
            retr = data.get("retrieval_strategy", "none")
            ev_strat = data.get("eval_strategy", "none")
            nr = data.get("n_refs", 0)
            
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

            if k == 1:
                acc = data.get("metrics", {}).get("accuracy")
                if acc is not None: acc = acc * 100
            else:
                pred_file_path = preds_dir / f.name
                acc = calc_top_k(pred_file_path, k)
                
            if acc is None: continue
            grouped.setdefault(clf, {}).setdefault(key, []).append(acc)

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
                    baseline_storage.setdefault(clf, {}).setdefault(ev_strat, []).append(base_acc)
            except Exception:
                pass

        if not grouped:
            continue

        all_strategies = set()
        for clf in grouped:
            all_strategies.update(grouped[clf].keys())
        all_strategies = sorted(list(all_strategies))

        valid_classifiers = [
            clf for clf, strats in grouped.items()
            if all(s in strats and len(strats[s]) > 0 for s in all_strategies)
        ]
        
        if not valid_classifiers:
            continue

        group_priority = {"CNN": 0, "Vision Transformer": 1, "Vision-Language Model": 2, "Foundation Model": 3, "Unknown": 4}
        valid_classifiers.sort(key=lambda c: (group_priority.get(get_classifier_name(c)[1], 4), c))
        x_indices = np.arange(len(valid_classifiers))
        
        out_dir = output_dir / method / args.split / f"ablation_{ablation_type}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        num_strategies = len(all_strategies)
        total_group_width = 0.8
        bar_width = total_group_width / num_strategies
        strat_color_map = {}
        
        for i, strat in enumerate(all_strategies):
            strat_means = [np.mean(grouped[clf][strat]) for clf in valid_classifiers]
            strat_stds = [np.std(grouped[clf][strat]) for clf in valid_classifiers]
            
            offsets = (i - (num_strategies - 1) / 2) * bar_width
            bar_x = x_indices + offsets
            
            rects = ax.bar(bar_x, strat_means, yerr=strat_stds, width=bar_width, label=strat, capsize=3)
            strat_color_map[strat] = rects[0].get_facecolor()
            
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=2.5, rotation=90)

        base_x, base_means = [], []
        for idx, clf in enumerate(valid_classifiers):
            clf_base_values = []
            for ev_mode in baseline_storage.get(clf, {}):
                clf_base_values.extend(baseline_storage[clf][ev_mode])
            if clf_base_values:
                base_x.append(idx)
                base_means.append(np.mean(clf_base_values))
                
        if base_means:
            for idx, b_mean in zip(base_x, base_means):
                start_x = idx - (total_group_width / 2)
                end_x = idx + (total_group_width / 2)
                lbl = "Baseline Performance" if idx == base_x[0] else ""
                ax.hlines(b_mean, xmin=start_x, xmax=end_x, colors="gray", linestyles="--", linewidth=1.5, label=lbl, zorder=3)

        clean_labels = [get_classifier_name(clf)[0] for clf in valid_classifiers]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(clean_labels, rotation=15, ha="right")
        ax.set_xlabel("Classifiers")
        ax.set_ylabel(f"Top-{k} Accuracy (%)")
        ax.set_title(f"Aggregated {ablation_type.capitalize()} Performance (Top-{k})")
        
        handles, labels = ax.get_legend_handles_labels()
        sorted_pairs = sorted(zip(labels, handles), key=lambda t: t[0])
        ax.legend([p[1] for p in sorted_pairs], [p[0] for p in sorted_pairs], loc="best")
        
        all_points = [m for c in valid_classifiers for s in all_strategies for m in grouped[c][s]] + base_means
        ax.set_ylim(bottom=max(0, min(all_points) - 5), top=min(100, max(all_points) + 12))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        fig.tight_layout()
        
        fig.savefig(out_dir / f"aggregated_ablation_{ablation_type}_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Table generation setup
        headers = ["Classifier"] + all_strategies
        if base_means:
            headers.append("Baseline")
            
        table_data = []
        row_max_indices = []  # Stores the column index of the maximum value for each row
        
        for row_idx, clf in enumerate(valid_classifiers):
            clf_name = get_classifier_name(clf)[0]
            row = [clf_name]
            
            row_vals = [np.mean(grouped[clf][strat]) for strat in all_strategies]
            highest_strat_idx = np.argmax(row_vals)
            row_max_indices.append(highest_strat_idx + 1)  # +1 offsets for classifier name
            
            for val in row_vals:
                row.append(f"{val:.1f}")
                
            if base_means:
                row.append(f"{base_means[base_x.index(row_idx)]:.1f}" if row_idx in base_x else "-")
            table_data.append(row)
            
        fig_tbl, ax_tbl = plt.subplots(figsize=(2 + 1.5 * len(headers), 1 + 0.3 * len(table_data)))
        ax_tbl.axis('off')
        
        tbl = ax_tbl.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.2)
        
        for i, strat in enumerate(all_strategies):
            cell = tbl[0, i + 1]
            cell.get_text().set_color(strat_color_map[strat])
            cell.get_text().set_weight('bold')
        if base_means:
            tbl[0, len(headers) - 1].get_text().set_color('gray')
            tbl[0, len(headers) - 1].get_text().set_weight('bold')
            
        # Apply row-by-row top target cell updates
        for row_idx, max_col_idx in enumerate(row_max_indices):
            target_cell = tbl[row_idx + 1, max_col_idx]
            target_cell.get_text().set_color('darkred')
            target_cell.get_text().set_weight('bold')
            target_cell.set_facecolor('#ffcccc')
            
        fig_tbl.savefig(out_dir / f"metrics_table_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig_tbl)

def plot_ablation_nrefs(results_dir: Path, output_dir: Path, method: str, args):
    suffix = f"tta_inference/results/{args.dataset}/{args.split}"
    res_dir = results_dir / method / suffix
    
    axis_values = [2, 4, 8, 16, 32, 64]
    if method == "ablation/adain_tta":
        template = "{cl}_adain_tta_zero_dino_nrefs{rfs}_seed71397589.json"
    elif method == "ablation/style_tta":
        template = "{cl}_style_tta_zero_dino_nrefs{rfs}_seed71397589.json"
        axis_values = axis_values[:4]
    elif method == "geometric_tta":
        template = "{cl}_geometric_zero_nviews{rfs}_seed71397589.json"
    else:
        return

    out_dir = output_dir / method / args.split / "ablation_nrefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for k in [1, 5]:
        metrics = {}
        baselines = {}
        
        for cl in ALL_CLASSIFIERS:
            for rfs in axis_values:
                f_name = template.format(cl=cl, rfs=rfs)
                f = res_dir / f_name
                if not f.exists(): continue
                    
                try:
                    data = load_json(f)
                    if k == 1:
                        acc = data.get("metrics", {}).get("accuracy")
                        if acc is not None: acc *= 100
                    else:
                        preds_path = results_dir / method / f"tta_inference/predictions/{args.dataset}/{args.split}" / f_name
                        acc = calc_top_k(preds_path, k)
                        
                    if acc is not None:
                        metrics.setdefault(cl, {})[str(rfs)] = acc
                        
                    if cl not in baselines:
                        base_path = get_baseline_results(f_name, k != 1, args.split)
                        base_acc = load_json(base_path).get("metrics", {}).get("accuracy") * 100 if k == 1 else calc_top_k(base_path, k)
                        baselines[cl] = base_acc
                except Exception:
                    continue

        classifiers = list(metrics.keys())
        if not classifiers: continue
        x_indices = np.arange(len(classifiers))
        
        # --- 1. Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Reduced width for tighter horizontal cluster spacing
        group_spread = 0.4 
        num_vals = len(axis_values)
        point_spacing = group_spread / num_vals
        blues_gradient = [plt.cm.Blues(0.3 + 0.7 * (i / (num_vals - 1))) for i in range(num_vals)]
        strat_colors = {}
        for i, rfs in enumerate(axis_values):
            means = [metrics[cl].get(str(rfs), 0.0) for cl in classifiers]
            offsets = (i - (len(axis_values) - 1) / 2) * point_spacing
            
            # Pass the generated gradient color to the scatter function
            paths = ax.scatter(x_indices + offsets, means, s=40, 
                            color=blues_gradient[i], label=str(rfs), zorder=4)
            strat_colors[str(rfs)] = blues_gradient[i]


        for idx, cl in enumerate(classifiers):
            if cl in baselines:
                lbl = "Baseline" if idx == 0 else ""
                ax.hlines(baselines[cl], xmin=idx - (group_spread / 2), xmax=idx + (group_spread / 2), 
                    colors="gray", linestyles="--", linewidth=1.5, label=lbl, zorder=3)

        clean_labels = [get_classifier_name(cl)[0] for cl in classifiers]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(clean_labels, rotation=15, ha="right")
        
        ax.set_ylabel(f"Top-{k} Accuracy")
        ax.set_title(f"nrefs Ablation (Top-{k})\n[Retrieval: DINO | Eval: Zero Filtering]")
        ax.legend(title="n_refs")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"aggregated_ablation_nrefs_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        # --- 2. Table Render ---
        headers = ["Classifier"] + [str(r) for r in axis_values] + ["Baseline"]
        table_data = []
        row_max_cols = []
        
        for r_idx, cl in enumerate(classifiers):
            row = [cl]
            row_vals = [metrics[cl].get(str(r), 0.0) for r in axis_values]
            row_max_cols.append(np.argmax(row_vals) + 1)
            
            row.extend([f"{v:.1f}" if v > 0 else "-" for v in row_vals])
            row.append(f"{baselines[cl]:.1f}" if cl in baselines else "-")
            table_data.append(row)
            
        fig_tbl, ax_tbl = plt.subplots(figsize=(2 + 1.2 * len(headers), 1 + 0.3 * len(table_data)))
        ax_tbl.axis('off')
        tbl = ax_tbl.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.2)
        
        for i, rfs in enumerate(axis_values):
            cell = tbl[0, i + 1]
            cell.get_text().set_color(strat_colors[str(rfs)])
            cell.get_text().set_weight('bold')
            
        for r_idx, max_col in enumerate(row_max_cols):
            cell = tbl[r_idx + 1, max_col]
            cell.get_text().set_color('darkred')
            cell.get_text().set_weight('bold')
            cell.set_facecolor('#ffcccc')
            
        fig_tbl.savefig(out_dir / f"metrics_table_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig_tbl)

def plot_all_ablation_nrefs(results_dir: Path, output_dir: Path, method: str, args, all_comparison: bool = False):
    methods_to_run = ["ablation/adain_tta", "ablation/style_tta", "geometric_tta", "hybrid_tta"] if all_comparison else [method]
    save_folder = "all_methods_comparison" if all_comparison else method
    out_dir = output_dir / save_folder / args.split / "ablation_nrefs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    method_configs = {
        "ablation/adain_tta": {"template": "{cl}_adain_tta_zero_dino_nrefs{rfs}_seed{seed}.json", "axis": [2, 4, 8, 16, 32, 64], "color": "tab:red", "label": "AdaIN"},
        "ablation/style_tta": {"template": "{cl}_style_tta_zero_dino_nrefs{rfs}_seed{seed}.json", "axis": [2, 4, 8, 16], "color": "tab:green", "label": "StyleID"},
        "geometric_tta": {"template": "{cl}_geometric_zero_nviews{rfs}_seed{seed}.json", "axis": [2, 4, 8, 16, 32, 64], "color": "tab:blue", "label": "Geometric(Crop/Flip)"}
    }

    for k in [1, 5]:
        # Store as: data[method][rfs] = [list_of_accuracies]
        data_store = {m: {r: [] for r in method_configs[m]["axis"]} for m in methods_to_run}
        
        for m in methods_to_run:
            cfg = method_configs[m]
            for cl in ALL_CLASSIFIERS:
                for rfs in cfg["axis"]:
                    acc = []
                    for seed  in ALL_SEEDS:
                        f = results_dir / m / f"tta_inference/results/{args.dataset}/{args.split}" / cfg["template"].format(cl=cl, rfs=rfs, seed=seed)
                        if not f.exists(): continue
                        
                        data = load_json(f)
                        acc.append(data.get("metrics", {}).get("accuracy") * 100 if k == 1 else calc_top_k(results_dir / m / f"tta_inference/predictions/{args.dataset}/{args.split}" / f.name, k))
                    if acc is not None:
                        acc_mean = np.mean(np.array(acc))
                        data_store[m][rfs].append(acc_mean)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for m in methods_to_run:
            cfg = method_configs[m]
            # Average over classifiers for each rfs
            x_vals = cfg["axis"]
            y_vals = [np.mean(data_store[m][r]) if data_store[m][r] else None for r in x_vals]
            
            # Filter None
            valid = [(x, y) for x, y in zip(x_vals, y_vals) if y is not None]
            if valid:
                ax.plot([v[0] for v in valid], [v[1] for v in valid], 
                        marker='o', linestyle='-', color=cfg["color"], label=cfg["label"])

        ax.set_xlabel("n_refs")
        ax.set_ylabel(f"Average Top-{k} Accuracy (%)")
        ax.set_title(f"Ablation n_refs (Averaged Over All Classifiers)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        fig.savefig(out_dir / f"line_plot_nrefs_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

def plot_ablation_nrefs_multi(results_dir, output_dir, strategy_keys, args):
    out_dir = output_dir / "aggregated_ablation" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for k in [1, 5]:
        results_store = {}
        baselines = {}
        
        for s_key in strategy_keys:
            cfg = TTA_STRATEGIES[s_key]
            results_store[s_key] = {}
            for cl in ALL_CLASSIFIERS:
                results_store[s_key][cl] = {}
                for rfs in cfg["axis"]:
                    accs = []
                    for seed in [DEFAULT_SEED]:
                        if s_key == "hybrid_tta":
                            real_rfs, geo_fac = rfs
                            f_name = cfg["template"].format(cl=cl, geo=geo_fac, rfs=real_rfs, seed=seed)
                        elif s_key == "ablation/adain_tta":
                            f_name = cfg["template"].format(cl=cl, eval="zero", retr="dino", rfs=rfs, seed=seed)
                        elif s_key == "ablation/style_tta":
                            f_name = cfg["template"].format(cl=cl, eval="vanilla", retr="dino", rfs=rfs, seed=seed)
                        elif s_key == "geometric_tta":
                            f_name = cfg["template"].format(cl=cl, eval="vanilla", rfs=rfs, seed=seed, retr="")
                        else:
                            f_name = cfg["template"].format(cl=cl, rfs=rfs, seed=seed)
                            
                        f = results_dir / s_key / f"tta_inference/results/{args.dataset}/{args.split}" / f_name
                      
                        if not f.exists(): continue
                        
                        data = load_json(f)
                        if k == 1:
                            acc = data.get("metrics", {}).get("accuracy", 0) * 100  
                        else: 
                            if s_key == "hybrid_tta":
                                acc = data.get("metrics", {}).get("top5_accuracy", 0) * 100 
                            else:
                                calc_top_k(results_dir / s_key / f"tta_inference/predictions/{args.dataset}/{args.split}" / f_name, k)
                        accs.append(acc)
                    if accs:
                        results_store[s_key][cl][str(rfs)] = np.mean(accs)
                
                if cl not in baselines:
                    base_path = get_baseline_results(f_name, k != 1, args.split)
                    baselines[cl] = load_json(base_path).get("metrics", {}).get("accuracy", 0) * 100 if k == 1 else calc_top_k(base_path, k)

        fig, ax = plt.subplots(figsize=(14, 7))
        classifiers = ALL_CLASSIFIERS
        x_indices = np.arange(len(classifiers))
        
        total_methods = len(strategy_keys)
        group_width = 0.6
        method_width = group_width / total_methods
        
        for s_idx, s_key in enumerate(strategy_keys):
            cfg = TTA_STRATEGIES[s_key]
            axis_vals = cfg["axis"]
            
            sub_step = method_width / len(axis_vals)
            
            for i, rfs in enumerate(axis_vals):
                means = [results_store[s_key].get(cl, {}).get(str(rfs), np.nan) for cl in classifiers]
                offset = (s_idx * method_width) - (group_width / 2) + (i * sub_step)
                
                color = plt.colormaps[cfg["color_shade"]](0.3 + 0.7 * (i / len(axis_vals)))
                ax.scatter(x_indices + offset, means, color=color, label=f"{cfg['label']} (n={rfs})", s=40, alpha=0.9, zorder=3)

        for idx, cl in enumerate(classifiers):
            ax.hlines(baselines.get(cl, 0), xmin=idx - group_width/2, xmax=idx + group_width/2, 
                      colors="gray", linestyles="--", linewidth=1.2, zorder=2)
        ax.set_ylim(30, 85)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([get_classifier_name(cl)[0] for cl in classifiers], rotation=15, ha="right")
        ax.set_ylabel(f"Top-{k} Accuracy (%)")
        ax.set_title(f"Ablation Study: Influence of N-Views on Top-{k} Accuracy\nDataset: ImageNet-R | Comparison: {', '.join([TTA_STRATEGIES[s]['label'] for s in strategy_keys])}")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(out_dir / f"comparison{len(strategy_keys)}_nrefs_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

def plot_all_ablation_retr(results_dir: Path, output_dir: Path, method: str, args, all_comparison: bool = False):
    methods_to_run = ["ablation/adain_tta"] if all_comparison else [method]
    
    save_folder = "all_methods_comparison" if all_comparison else method
    out_dir = output_dir / save_folder / args.split / "ablation_retr"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    method_configs = {
        "ablation/adain_tta": {
            "template": "{cl}_adain_tta_zero_{strat}_nrefs32_seed71397589.json",
            "axis": ["dino", "random", "balanced_random"],
            "cmap": plt.cm.Reds,
            "label": "AdaIN"
        }
    }

    for k in [1, 5]:
        metrics = {}
        baselines = {}
        
        for m in methods_to_run:
            cfg = method_configs[m]
            suffix = f"tta_inference/results/{args.dataset}/{args.split}"
            res_dir = results_dir / m / suffix
            
            for cl in ALL_CLASSIFIERS:
                for strat in cfg["axis"]:
                    f_name = cfg["template"].format(cl=cl, strat=strat)
                    f = res_dir / f_name
                    if not f.exists(): continue
                        
                    try:
                        data = load_json(f)
                        if k == 1:
                            acc = data.get("metrics", {}).get("accuracy")
                            if acc is not None: acc *= 100
                        else:
                            preds_path = results_dir / m / f"tta_inference/predictions/{args.dataset}/{args.split}" / f_name
                            acc = calc_top_k(preds_path, k)
                            
                        if acc is not None:
                            metrics.setdefault(cl, {})[(m, strat)] = acc
                            
                        if cl not in baselines:
                            base_path = get_baseline_results(f_name, k != 1, args.split)
                            base_acc = load_json(base_path).get("metrics", {}).get("accuracy") * 100 if k == 1 else calc_top_k(base_path, k)
                            baselines[cl] = base_acc
                    except Exception:
                        continue

        classifiers = list(metrics.keys())
        if not classifiers: continue
        x_indices = np.arange(len(classifiers))
        
    # --- 1. Plotting (Bar Chart Version) ---
        fig, ax = plt.subplots(figsize=(12, 6))
        group_spread = 0.6 if all_comparison else 0.4
        
        all_variations = []
        for m in methods_to_run:
            for strat in method_configs[m]["axis"]:
                all_variations.append((m, strat))
                
        total_plots = len(all_variations)
        # Bar width ensures they fit within the group_spread
        bar_width = group_spread / max(1, total_plots)
        
        for idx, (m, strat) in enumerate(all_variations):
            means = [metrics[cl].get((m, strat), 0.0) for cl in classifiers]
            # Calculate offset: center the whole group, then position each bar
            offsets = (idx - (total_plots - 1) / 2) * bar_width
            
            ax_len = len(method_configs[m]["axis"])
            curr_idx = method_configs[m]["axis"].index(strat)
            color_grade = method_configs[m]["cmap"](0.3 + 0.7 * (curr_idx / max(1, ax_len - 1)))
            
            lbl = f"{method_configs[m]['label']}-{strat}"
            # Use ax.bar instead of ax.scatter
            ax.bar(x_indices + offsets, means, width=bar_width * 0.9, color=color_grade, label=lbl, zorder=4)

        # Baseline lines (adjusting width to match group_spread)
        for idx, cl in enumerate(classifiers):
            if cl in baselines:
                lbl = "Baseline" if idx == 0 else ""
                ax.hlines(baselines[cl], xmin=idx - (group_spread / 2), xmax=idx + (group_spread / 2), 
                          colors="gray", linestyles="--", linewidth=1.5, label=lbl, zorder=3)
        clean_labels = [get_classifier_name(cl)[0] for cl in classifiers]
        ax.set_xticks(x_indices)
        ax.set_xticklabels(clean_labels, rotation=15, ha="right")
        ax.set_ylabel(f"Top-{k} Accuracy")
        ax.set_title(f"Retrieval Strategy Ablation Comparison (Top-{k})\nDataset: ImageNet-R | N-Views: 32 (Original + 31 Stylised)")
        
        ax.legend(title="Methods & Strategies", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"aggregated_ablation_retr_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig)

        headers = ["Classifier"] + [f"{method_configs[m]['label']}-{s}" for m, s in all_variations] + ["Baseline"]
        table_data = []
        row_max_cols = []
        
        for r_idx, cl in enumerate(classifiers):
            row = [get_classifier_name(cl)[0]]
            row_vals = [metrics[cl].get((m, s), 0.0) for m, s in all_variations]
            row_max_cols.append(np.argmax(row_vals) + 1)
            
            row.extend([f"{v:.1f}" if v > 0 else "-" for v in row_vals])
            row.append(f"{baselines[cl]:.1f}" if cl in baselines else "-")
            table_data.append(row)
            
        fig_tbl, ax_tbl = plt.subplots(figsize=(2 + 1.1 * len(headers), 1 + 0.3 * len(table_data)))
        ax_tbl.axis('off')
        tbl = ax_tbl.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.2)
        
        for idx, (m, strat) in enumerate(all_variations):
            ax_len = len(method_configs[m]["axis"])
            curr_idx = method_configs[m]["axis"].index(strat)
            color_grade = method_configs[m]["cmap"](0.4 + 0.6 * (curr_idx / max(1, ax_len - 1)))
            
            cell = tbl[0, idx + 1]
            cell.get_text().set_color(color_grade)
            cell.get_text().set_weight('bold')
            
        for r_idx, max_col in enumerate(row_max_cols):
            cell = tbl[r_idx + 1, max_col]
            cell.get_text().set_color('darkred')
            cell.get_text().set_weight('bold')
            cell.set_facecolor('#ffcccc')
            
        fig_tbl.savefig(out_dir / f"metrics_table_top{k}.png", bbox_inches='tight', dpi=300)
        plt.close(fig_tbl)


def plot_ablation_retrieval():
    return

def plot_nrefs_sweep(results_dir: Path, output_dir: Path, args, method: str, best_retrieval="dino", best_eval="zero"):
    res_suffix = method 
    abl_dir = results_dir / res_suffix
    if (results_dir / res_suffix / "tta_inference" / "results").exists():
        abl_dir = results_dir / res_suffix / "tta_inference" / "results" / args.dataset / args.split
    res_suffix = res_suffix.replace("/", "_")
    files = list(abl_dir.glob("**/*.json"))
    if not files:
        return

    grouped: Dict[Tuple[int, str], List[float]] = {}
    for f in files:
        data = load_json(f)
        retr = data.get("retrieval_strategy", "none")
        ev_strat = data.get("eval_strategy", "none")
        
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
        return

    all_nrefs = sorted(set(k[0] for k in grouped))
    
    # --- Filter: Classifiers must have complete trajectories across all resolutions ---
    all_raw_classifiers = set(k[1] for k in grouped)
    valid_classifiers = [
        clf for clf in all_raw_classifiers
        if all((nr, clf) in grouped and len(grouped[(nr, clf)]) > 0 for nr in all_nrefs)
    ]

    if not valid_classifiers:
        return

    group_priority = {"CNN": 0, "Vision Transformer": 1, "Vision-Language Model": 2, "Foundation Model": 3, "Unknown": 4}
    valid_classifiers.sort(key=lambda c: (group_priority.get(get_classifier_name(c)[1], 4), c))

    fig, ax = plt.subplots(figsize=(8, 5))
    
    for clf in valid_classifiers:
        means = [np.mean(grouped[(nr, clf)]) for nr in all_nrefs]
        stds = [np.std(grouped[(nr, clf)]) for nr in all_nrefs]
        
        clean_name, _ = get_classifier_name(clf)
        ax.errorbar(all_nrefs, means, yerr=stds, marker="o", markersize=5, linewidth=1.5, label=clean_name, capsize=4)

    ax.set_xlabel("Number of Style References ($n_{refs}$)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title(f"Style-Transfer TTA Scale Sweep ({best_retrieval.upper()} + {best_eval.capitalize()})", fontsize=11, fontweight='bold')
    
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(all_nrefs)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    
    # Auto-Sort Legend Alphabetically by Clean Display Name
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda t: t[0])
    ax.legend([p[1] for p in sorted_pairs], [p[0] for p in sorted_pairs], fontsize=9, loc="best")
    
    fig.tight_layout()
    out_dir = output_dir / method / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{res_suffix}_nrefs_sweep_filtered.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_all_topk_metrics(results_dir: Path, dataset, split):
    methods = {
        "AdaIN": "ablation/adain_tta",
        "RetriStyle": "ablation/style_tta",
        "Geometric": "geometric_tta"
    }
    templates = {
        "AdaIN": "{cl}_adain_tta_zero_dino_nrefs64_seed71397589.json",
        "RetriStyle": "{cl}_style_tta_zero_dino_nrefs16_seed71397589.json",
        "Geometric": "{cl}_geometric_zero_nviews64_seed71397589.json"
    }

    # Data structure: { "ClassifierName": { "Method": (t1, t5), "Base": (b1, b5) } }
    results = {get_classifier_name(cl)[0]: {"Base": (0, 0), "AdaIN": (0,0), "RetriStyle": (0,0), "Geometric": (0,0)} 
               for cl in ALL_CLASSIFIERS}

    for method_name, folder in methods.items():
        for cl in ALL_CLASSIFIERS:
            cl_name = get_classifier_name(cl)[0]
            template = templates[method_name]
            f_name = template.format(cl=cl)
            f = results_dir / folder / f"tta_inference/results/{dataset}/{split}" / f_name
            
            if not f.exists(): continue
            
            t1 = (load_json(f).get("metrics", {}).get("accuracy", 0)) * 100
            preds_p = results_dir / folder / f"tta_inference/predictions/{dataset}/{split}" / f_name
            t5 = calc_top_k(preds_p, 5)
            
            results[cl_name][method_name] = (t1, t5)
            
            # Baseline (only set once per classifier)
            if results[cl_name]["Base"] == (0,0):
                base_p = get_baseline_results(f_name, False, split)
                b1 = load_json(base_p).get("metrics", {}).get("accuracy", 0) * 100 if base_p.exists() else 0
                b5 = calc_top_k(get_baseline_results(f_name, True, split), 5)
                results[cl_name]["Base"] = (b1, b5)

    # Plotting
    for k_type in ["Top-1", "Top-5"]:
        idx = 0 if k_type == "Top-1" else 1
        filename = f"{k_type.lower().replace('-', '')}_accuracy.png"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(results))
        width = 0.2
        
        for i, method in enumerate(["Base", "AdaIN", "RetriStyle", "Geometric"]):
            vals = [results[cl][method][idx] for cl in results]
            ax.bar(x + (i - 1.5) * width, vals, width, label=method)

        ax.set_title(f"Comparison: {k_type} Accuracy")
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(results.keys(), rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300) # Saves the file at 300 DPI
        plt.close(fig) # Closes the figure to free up memory
        print(f"Saved: {filename}")

if __name__ == "__main__":
    res = Path("/home/stud/nemmler/style_tta/results")

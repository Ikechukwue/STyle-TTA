from code.config.helpers import load_json, get_names, get_top_k, calc_top_k, get_classifier_name
from code.config.constants import ALL_CLASSIFIERS, DOMAIN_METRICS,TTA_STRATEGIES, DEFAULT_SEED
from pathlib import Path
import scipy.stats as stats
import numpy as np 
def print_domain_gap(domain_gap_path):
    data = load_json(domain_gap_path / "full_cross_domain_gap.json")
    metrics_to_show = ["mmd", "wasserstein", "kl_symmetric"]
    
    report_data = {}

    for cls in data["ood"]:
        print(f"\n=== {cls} ===")
        print(f"{'Metric':<20} | {'Baseline (mean ± std)':<25} | {'OOD (mean ± std)':<25} | {'Increase':<10}")
        print("-" * 88)
        
        report_data[cls] = {}

        for metric in metrics_to_show:
            if cls == "gram_distance":            
                metric = cls        
            b_mean = data["baseline"][cls].get(f"{metric}_mean")
            b_std = data["baseline"][cls].get(f"{metric}_std")
            o_mean = data["ood"][cls].get(f"{metric}_mean")
            o_std = data["ood"][cls].get(f"{metric}_std")

            if None in (b_mean, b_std, o_mean, o_std):
                continue

            increase = ((o_mean - b_mean) / b_mean) * 100
            baseline_str = f"{b_mean:.4f} ± {b_std:.4f}"
            ood_str = f"{o_mean:.4f} ± {o_std:.4f}"

            print(f"{metric:<20} | {baseline_str:<25} | {ood_str:<25} | {'+' if increase > 0 else ''}{increase:.1f}%")
            
            report_data[cls][metric] = {
                "baseline_mean": b_mean,
                "baseline_std": b_std,
                "ood_mean": o_mean,
                "ood_std": o_std,
                "increase_pct": increase
            }
            
            if cls == "gram_distance":
                break
                
    return report_data


def print_domain_gap_class(domain_path):
    n_classes = 200
    global_data = load_json(domain_path / "full_cross_domain_gap.json")
    results = {str(n): {} for n in range(n_classes)}
    metrics_to_show = ["mmd", "wasserstein", "kl_symmetric"]

    for cls in global_data["ood"]:
        file_path = domain_path / f"{cls}_domain_gap.json"
        if not file_path.exists():
            continue
            
        class_data = load_json(file_path)
        for n in results:
            target_metric = cls if cls == "gram_distance" else None
            
            if target_metric is None:
                for metric in metrics_to_show:
                    b_mean = class_data["baseline_gap"]["per_class"][n].get(metric)
                    o_mean = class_data["domain_gap"]["per_class"][n].get(metric)
                    
                    if None not in (b_mean, o_mean) and b_mean != 0:
                        increase = ((o_mean - b_mean) / b_mean) * 100
                        results[n][metric] = increase
            else:
                b_mean = class_data["baseline_gap"]["per_class"][n].get(cls)
                o_mean = class_data["domain_gap"]["per_class"][n].get(cls)
                
                if None not in (b_mean, o_mean) and b_mean != 0:
                    increase = ((o_mean - b_mean) / b_mean) * 100
                    results[n][cls] = increase

    valid_results = {
        n: metrics for n, metrics in results.items() 
        if "gram_distance" in metrics and any(m in metrics for m in metrics_to_show)
    }

    if not valid_results:
        print("\n[Error] No classes found with complete metric pairings.")
        return {}

    sorted_by_gram = sorted(
        valid_results.items(), 
        key=lambda item: item[1].get("gram_distance", -float("inf")), 
        reverse=True
    )

    print("\n" + "="*95)
    print("LEADERBOARD: TOP 10 HIGHEST TEXTURE DEVIATIONS (SORTED BY GRAM INCREASE)")
    print("="*95)
    print(f"{'Class ID':<10} | {'Gram Dist %':<15} | {'MMD %':<15} | {'Wasserstein %':<15} | {'KL %':<15}")
    print("-" * 95)
    
    name_list = get_names()
    for n, metrics in sorted_by_gram[:10]:
        gram_str = f"+{metrics['gram_distance']:.1f}%"
        mmd_str = f"+{metrics['mmd']:.1f}%" if "mmd" in metrics else "N/A"
        wass_str = f"+{metrics['wasserstein']:.1f}%" if "wasserstein" in metrics else "N/A"
        kl_str = f"+{metrics['kl_symmetric']:.1f}%" if "kl_symmetric" in metrics else "N/A"
        class_name = name_list[int(n)]
        print(f"Class {class_name:<8} | {gram_str:<15} | {mmd_str:<15} | {wass_str:<15} | {kl_str:<15}")
    print("-" * 95)

    print("\n" + "="*95)
    print(f"GLOBAL TEXTURE-BIAS ANALYSIS (CORRELATION ACROSS ALL {len(valid_results)} EVALUATED CLASSES)")
    print("="*95)
    print(f"{'Feature Space Metric':<25} | {'Pearson r':<15} | {'Spearman ρ':<15} | {'Statistical Significance'}")
    print("-" * 95)

    correlations = {}

    for metric in metrics_to_show:
        pairs = [(metrics["gram_distance"], metrics[metric]) for metrics in valid_results.values() if metric in metrics]
        if len(pairs) < 3:
            continue
            
        g_vec, m_vec = zip(*pairs)
        pearson_r, p_pearson = stats.pearsonr(g_vec, m_vec)
        spearman_rho, p_spearman = stats.spearmanr(g_vec, m_vec)
        
        sig = "Highly Significant" if p_spearman < 0.001 else "Significant" if p_spearman < 0.05 else "Not Significant"
        print(f"{metric:<25} | {pearson_r:<15.4f} | {spearman_rho:<15.4f} | {sig} (p={p_spearman:.2e})")
        
        correlations[metric] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(p_pearson),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(p_spearman),
            "significance": sig
        }
    print("="*95 + "\n")

    return {
        "class_metrics": valid_results,
        "correlations": correlations
    }

def get_domain_gap_class(domain_path=Path("./results/domain_gap/feature_space/test_r")):
    n_classes = 200
    global_data = load_json(domain_path / "full_cross_domain_gap.json")
    results = {str(n): {} for n in range(n_classes)}
    metrics_to_show = ["mmd", "wasserstein", "kl_symmetric"]

    for cls in global_data["ood"]:
        file_path = domain_path / f"{cls}_domain_gap.json"
        if not file_path.exists():
            continue
        
        class_data = load_json(file_path)
        for n in results:
            target_metric = cls if cls == "gram_distance" else None
            
            if target_metric is None:
                results[n][cls] = {}
                for metric in metrics_to_show:
                    b_mean = class_data["baseline_gap"]["per_class"][n].get(metric)
                    o_mean = class_data["domain_gap"]["per_class"][n].get(metric)
                    
                    if None not in (b_mean, o_mean) and b_mean != 0:
                        increase = o_mean - b_mean
                        results[n][cls][metric] = increase
            else:
                b_mean = class_data["baseline_gap"]["per_class"][n].get(cls)
                o_mean = class_data["domain_gap"]["per_class"][n].get(cls)
                
                if None not in (b_mean, o_mean) and b_mean != 0:
                    increase = o_mean - b_mean
                    results[n][cls] = increase

    return results

def get_class_acc():
    results_dir = Path("./results/geometric_tta/tta_inference/predictions/imagenet")
    template = TTA_STRATEGIES["geometric_tta"]["template"]
    datasets = ["val@test_r", "test_r"]

    def get_ytrue_mask(path):
        data = load_json(path)
        final_indices, _ = get_top_k(data, 1)
        y_true = np.array([sample["y_true"] for sample in data["predictions"]])
        correct_mask = np.any(final_indices == y_true[:, None], axis=-1)
        return correct_mask, y_true

    class_acc = {}
    for cls in ALL_CLASSIFIERS:
        temp = template.format(cl=cls, rfs=1, seed=DEFAULT_SEED)
        base_correct, base_true = get_ytrue_mask(results_dir / datasets[0] / temp)
        ood_correct, ood_true = get_ytrue_mask(results_dir / datasets[1] / temp)
        class_acc[cls] = {}

        for c in np.unique(base_true):
            base_mask = (base_true == c)
            ood_mask = (ood_true == c)
            base_acc = np.mean(base_correct[base_mask])
            ood_acc = np.mean(ood_correct[ood_mask])
            delta_acc = (base_acc - ood_acc) / base_acc
            
            class_acc[cls][str(c)] = {
                "base_acc": base_acc,
                "ood_acc": ood_acc,
                "delta_acc": delta_acc
            }
    return class_acc


def accuracy_metric_correlation(domain_path):
    global_data = load_json(domain_path / "full_cross_domain_gap.json")
    class_metrics = get_domain_gap_class(domain_path)
    metrics_to_show = ["mmd", "wasserstein", "kl_symmetric", "gram_distance"]
    
    class_acc = get_class_acc()
    correlations = {}

    for cls in ALL_CLASSIFIERS:
        correlations[cls] = {}
        for metric in metrics_to_show:
            if metric == "gram_distance":
                class_metric = [class_metrics[k][metric] for k in class_metrics]
            else:
                class_metric = [class_metrics[k][cls][metric] for k in class_metrics]

            deltas = [class_acc[cls][k]["delta_acc"] for k in class_acc[cls]]

            pearson_r, p_pearson = stats.pearsonr(class_metric, deltas)
            spearman_rho, p_spearman = stats.spearmanr(class_metric, deltas)

            correlations[cls][metric] = {
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
                "p_spearman": p_spearman
            }
    return correlations

def print_correlation_report(correlations: dict):
    """
    Takes the correlations dictionary output and prints an analysis table comparing all classifiers.
    """
    metrics = ["gram_distance", "mmd", "wasserstein", "kl_symmetric"]
    
    print("\n" + "=" * 115)
    print(f"{'CLASSIFIER TEXTURE-BIAS & DISTANCE CORRELATION REPORT':^115}")
    print("=" * 115)
    
    for classifier, metric_data in correlations.items():
        print(f"\n[Classifier]: {classifier}")
        print(f"{'Metric / Feature Space':<25} | {'Pearson r':<12} | {'Spearman ρ':<12} | {'p-value':<10} | {'Interpretation'}")
        print("-" * 115)
        
        for metric in metrics:
            stats_dict = metric_data.get(metric)
            if not stats_dict:
                continue
                
            r = stats_dict["pearson_r"]
            rho = stats_dict["spearman_rho"]
            p = stats_dict["p_spearman"]
            
            # Format p-value significance flag
            sig_flag = " (*)" if p < 0.05 else ""
            p_str = f"{p:.2e}{sig_flag}"
            
            # Interpret directional trend toward accuracy drop (delta_acc)
            if rho > 0.5 and p < 0.05:
                interpretation = "Strong texture bias driving error"
            elif rho > 0.2 and p < 0.05:
                interpretation = "Moderate texture bias"
            elif rho < -0.2 and p < 0.05:
                interpretation = "Inverse effect"
            else:
                interpretation = "No statistically significant texture bias"
                
            print(f"{metric:<25} | {r:<12.4f} | {rho:<12.4f} | {p_str:<10} | {interpretation}")
        print("-" * 115)
    print("(*) Indicates statistical significance at p < 0.05\n")

import matplotlib.pyplot as plt
from pathlib import Path

def plot_class_metric_variance(class_metrics, output_dir="figures/domain_comparison"):
    metrics = ["mmd", "wasserstein", "kl_symmetric"]
    
    # Identify classifiers present in class_metrics
    sample_class = next(iter(class_metrics.values()))
    classifiers = [cls for cls in sample_class if isinstance(sample_class[cls], dict)]

    for cls in classifiers:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        fig.suptitle(f"Per-Class Metric Variance: {get_classifier_name(cls)[0]}", fontsize=14)

        classes = []
        metric_values = {m: [] for m in metrics}

        # Extract values per class
        for class_id, cls_data in class_metrics.items():
            if cls in cls_data:
                classes.append(int(class_id))
                for m in metrics:
                    metric_values[m].append(cls_data[cls].get(m, None))

        # Plot each metric subplot
        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            mean_val = np.nanmean(metric_values[metric],)
            std_val = np.nanstd(metric_values[metric],)

            # Center line and variance span
            ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.5, label="Mean")
            ax.axhspan(mean_val - std_val, mean_val + std_val, color="red", alpha=0.15, label=r"$\pm 1\sigma$ Range")
            cv = std_val / mean_val if mean_val != 0 else 0
            textstr = f"$\sigma$: {std_val:.2f}\n$CV$: {cv:.2f}\nRange: {np.nanmax(metric_values[metric],) - np.nanmin(metric_values[metric],):.1f}%"

            # Sort y-values to show steep variance curve across classes
            sorted_indices = np.argsort(metric_values[metric],)
            sorted_classes = np.array(classes)[sorted_indices]
            sorted_y = np.array(metric_values[metric])[sorted_indices]

            # Place text box in upper left
            ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.scatter(range(len(sorted_y)), sorted_y, alpha=0.7, s=25, c="tab:blue")
            ax.set_title(DOMAIN_METRICS[metric])
            ax.set_xlabel("Classes (Sorted Low to High)")
            ax.set_ylabel("Domain Gap Increase")
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path / f"{cls}_variance_scatter.png", dpi=300)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    domain_path = Path("/home/stud/nemmler/retristyle/results/domain_gap/feature_space/test_r")

    #print_domain_gap(domain_path)
    #print_domain_gap_class(domain_path)
    #corr = accuracy_metric_correlation(domain_path)
    #print_correlation_report(corr)
    domain_dict = get_domain_gap_class(domain_path)
    domain_dict = plot_class_metric_variance(domain_dict)

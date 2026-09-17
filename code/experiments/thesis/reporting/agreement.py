import re
from pathlib import Path
import numpy as np
from code.config.constants import ALL_CLASSIFIERS
from code.config.helpers import load_json
import json
from tqdm import tqdm 
def get_top_k(results: dict, k: int = 5):
    y_pred_matrix = np.array(
        [sample["y_pred"] for sample in results["predictions"]]
    )
    num_samples = y_pred_matrix.shape[0]
    num_classes = y_pred_matrix.shape[1]

    k = min(k, num_classes)

    if k == 1:
        final_indices = np.argmax(y_pred_matrix, axis=-1)[:, None]
        final_confidences = np.take_along_axis(
            y_pred_matrix, final_indices, axis=-1
        )
        return final_indices, final_confidences

    top_k_unsorted_indices = np.argpartition(y_pred_matrix, -k, axis=-1)[
        :, -k:
    ]

    row_indices = np.arange(num_samples)[:, None]
    top_k_unsorted_values = y_pred_matrix[row_indices, top_k_unsorted_indices]

    sort_args = np.argsort(-top_k_unsorted_values, axis=-1)
    final_indices = np.take_along_axis(
        top_k_unsorted_indices, sort_args, axis=-1
    )
    final_confidences = np.take_along_axis(
        top_k_unsorted_values, sort_args, axis=-1
    )

    return final_indices, final_confidences


def calc_agreement(orig_pred, pred_path):
    preds_data = load_json(pred_path)

    results = {
        "top1_agreement_samples": [],
        "top5_exact_order_agreement_samples": [],
        "top1_rate": 0.0,
        "top5_any_order_rate": 0.0,
        "top5_exact_order_rate": 0.0,
        "per_class": {}
    }

    orig_sorted = {
        "predictions": sorted(
            orig_pred["predictions"], key=lambda x: x["sample_idx"]
        )
    }
    later_sorted = {
        "predictions": sorted(
            preds_data["predictions"], key=lambda x: x["sample_idx"]
        )
    }

    orig_indices, _ = get_top_k(orig_sorted, k=5)
    later_indices, _ = get_top_k(later_sorted, k=5)

    orig_top1 = orig_indices[:, 0]
    later_top1 = later_indices[:, 0]
    top1_matches = orig_top1 == later_top1

    top5_exact_matches = np.all(orig_indices == later_indices, axis=1)

    top5_overlap_rates = [
        len(np.intersect1d(orig_row, later_row)) / 5.0
        for orig_row, later_row in zip(orig_indices, later_indices)
    ]

    # Initialize per-class tracking accumulators
    class_stats = {}

    for i in range(len(later_sorted["predictions"])):
        sample = later_sorted["predictions"][i]
        sample_id = sample["sample_idx"]
        
        # Extract ground truth target class integer
        y_true = sample["y_true"]
        if isinstance(y_true, list):
            y_true = y_true[0]

        if y_true not in class_stats:
            class_stats[y_true] = {"total": 0, "top1_matches": 0, "top5_exact": 0, "top5_overlaps": []}

        class_stats[y_true]["total"] += 1
        class_stats[y_true]["top5_overlaps"].append(top5_overlap_rates[i])

        if top1_matches[i]:
            results["top1_agreement_samples"].append(sample_id)
            class_stats[y_true]["top1_matches"] += 1
        if top5_exact_matches[i]:
            results["top5_exact_order_agreement_samples"].append(sample_id)
            class_stats[y_true]["top5_exact"] += 1

    total_samples = len(later_sorted["predictions"])
    if total_samples > 0:
        results["top1_rate"] = round(
            (len(results["top1_agreement_samples"]) / total_samples) * 100, 2
        )
        results["top5_any_order_rate"] = round(
            float(np.mean(top5_overlap_rates)) * 100, 2
        )
        results["top5_exact_order_rate"] = round(
            (len(results["top5_exact_order_agreement_samples"]) / total_samples)
            * 100,
            2,
        )

        # Compute final percentages per target class
        for c, stats in class_stats.items():
            count = stats["total"]
            results["per_class"][str(c)] = {
                "count": count,
                "top1_rate": round((stats["top1_matches"] / count) * 100, 2),
                "top5_any_order_rate": round(float(np.mean(stats["top5_overlaps"])) * 100, 2),
                "top5_exact_order_rate": round((stats["top5_exact"] / count) * 100, 2),
            }

    return results

def agreement(
    baseline_dir, results_dir, best_settings, dataset="imagenet", split="test_r"
):
    results = {}
    eval_strat = best_settings["eval_strat"]
    method = best_settings["method"]
    retr_strat = best_settings["retr_strat"]
    results_path = Path(results_dir)

    for cl in tqdm(ALL_CLASSIFIERS, desc="Classifiers:"):
        baseline_path = (
            Path(baseline_dir)
            / f"tta_inference/predictions/{dataset}/{split}/{cl}_geometric_vanilla_nviews1_seed71397589.json"
        )

        if not baseline_path.exists():
            print(f"No baseline found for classifier: {cl}")
            continue

        baseline_data = load_json(baseline_path)
        results[cl] = {}
        prefix = f"{cl}_{method}_{eval_strat}{retr_strat}"
        print(prefix + " and " + str(results_path))
        pattern = f"{prefix}*.json"
        files = list(results_path.glob(pattern))
        if not files:
            print(f"No files found for {results_path} and pattern {pattern}")
            continue

        for f in tqdm(files, desc="Files", leave=False):
            agree = calc_agreement(baseline_data, f)
            if not agree:
                continue
            remainder = f.stem.replace(prefix, "")

            match = re.search(r"\d+", remainder)
            view_key = f"k_{match.group(0)}" if match else f.stem

            results[cl][view_key] = {
                "top1": agree["top1_rate"],
                "top5_any_order": agree["top5_any_order_rate"],
                "top5_exact_order": agree["top5_exact_order_rate"],
                "per_class": agree["per_class"],
            }
    return results

def analyze_and_report_agreement(
    baseline_dir,
    results_dir,
    best_settings,
    output_filepath="results/tta_agreement_summary.json",
    dataset="imagenet",
    split="test_r",
    overwrite=False,
    report=True,
):
    output_path = Path(output_filepath)

    if output_path.exists() and not overwrite:
        print(f"Loading cached agreement statistics from: {output_path}")
        with open(output_path, "r") as f:
            results = json.load(f)
    else:
        print("Cache miss. Computing TTA agreements across all classifiers...")
        results = agreement(
            baseline_dir, results_dir, best_settings, dataset, split
        )
        if results:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Fresh results successfully saved to: {output_path}")
        else:
            print("Nothing found")
            return results
    if not results:
        print("Warning: No classifier data found to analyze.")
        return results

    all_k_keys = set()
    for cl_data in results.values():
        all_k_keys.update(cl_data.keys())

    def extract_numeric_sort_key(key_string):
        match = re.search(r"\d+", key_string)
        return int(match.group(0)) if match else key_string

    sorted_k_keys = sorted(list(all_k_keys), key=extract_numeric_sort_key)
    metrics_to_report = ["top1", "top5_any_order", "top5_exact_order"]
    if report:

        for metric in metrics_to_report:
            print("\n" + "=" * 75)
            print(
                f"TTA CONSISTENCY MATRIX ({metric.upper()}) ({dataset.upper()} - {split.upper()})  -  {best_settings["method"].upper()}"
            )
            print("=" * 75)

            header = f"{'Classifier Head / Architecture':<35}" + "".join(
                f"{k:>10}" for k in sorted_k_keys
            )
            print(header)
            print("-" * len(header))

            k_column_totals = {k: [] for k in sorted_k_keys}
            classifier_overall_means = {}

            for cl, k_dict in results.items():
                row_str = f"{cl:<35}"
                cl_rates = []
                for k in sorted_k_keys:
                    metric_entry = k_dict.get(k, None)

                    if isinstance(metric_entry, dict):
                        rate = metric_entry.get(metric, None)
                    elif isinstance(metric_entry, (int, float)):
                        rate = metric_entry if metric == "top1" else None
                    else:
                        rate = None

                    if rate is not None:
                        row_str += f"{rate:>9}%"
                        k_column_totals[k].append(rate)
                        cl_rates.append(rate)
                    else:
                        row_str += f"{'-':>10}"
                print(row_str)

                if cl_rates:
                    classifier_overall_means[cl] = np.mean(cl_rates)

            print("-" * len(header))

            avg_row = f"{'MACRO AVERAGE CONSENSUS':<35}"
            for k in sorted_k_keys:
                if k_column_totals[k]:
                    avg_row += f"{np.mean(k_column_totals[k]):>9.2f}%"
                else:
                    avg_row += f"{'-':>10}"
            print(avg_row)
            print("=" * 75)

            print(f"\nINSIGHTS AND STANDOUT STATISTICS ({metric.upper()}):")
            print("-----------------------------------")

            if classifier_overall_means:
                most_robust = max(
                    classifier_overall_means, key=classifier_overall_means.get
                )
                least_robust = min(
                    classifier_overall_means, key=classifier_overall_means.get
                )

                print(
                    f"Most Robust Model:   {most_robust} "
                    f"({classifier_overall_means[most_robust]:.2f}% average baseline agreement)"
                )
                print(
                    f"Most Volatile Model: {least_robust} "
                    f"({classifier_overall_means[least_robust]:.2f}% average baseline agreement)"
                )

            if len(sorted_k_keys) >= 2:
                start_k, end_k = sorted_k_keys[0], sorted_k_keys[-1]
                highest_decay = -999
                worst_degrading_model = None

                for cl, k_dict in results.items():
                    start_entry = k_dict.get(start_k)
                    end_entry = k_dict.get(end_k)

                    if isinstance(start_entry, dict) and isinstance(end_entry, dict):
                        start_rate = start_entry.get(metric)
                        end_rate = end_entry.get(metric)
                    elif isinstance(start_entry, (int, float)) and isinstance(
                        end_entry, (int, float)
                    ):
                        start_rate = start_entry if metric == "top1" else None
                        end_rate = end_entry if metric == "top1" else None
                    else:
                        start_rate = None
                        end_rate = None

                    if start_rate is not None and end_rate is not None:
                        decay = start_rate - end_rate
                        if decay > highest_decay:
                            highest_decay = decay
                            worst_degrading_model = cl

                if worst_degrading_model and highest_decay != -999:
                    print(
                        f"Steepest Decay:      {worst_degrading_model} "
                        f"lost {highest_decay:.2f}% agreement going from {start_k} to {end_k}"
                    )
            print("=" * 75 + "\n")

    return results

def map_accuracy_with_agreement(agreement_json_path, baseline_dir, results_dir, best_settings, dataset="imagenet", split="test_r"):
    with open(agreement_json_path, "r") as f:
        agreement_data = json.load(f)
        
    eval_strat = best_settings["eval_strat"]
    method = best_settings["method"]
    retr_strat = best_settings["retr_strat"]
    results_path = Path(results_dir)
    
    mapped_analysis = {}

    for cl, k_dict in agreement_data.items():
        baseline_path = (
            Path(baseline_dir)
            / f"tta_inference/predictions/{dataset}/{split}/{cl}_geometric_vanilla_nviews1_seed71397589.json"
        )
        if not baseline_path.exists():
            continue
            
        baseline_data = load_json(baseline_path)
        orig_indices, _ = get_top_k({
            "predictions": sorted(baseline_data["predictions"], key=lambda x: x["sample_idx"])
        }, k=1)
        
        # Get ground truth array
        sorted_preds = sorted(baseline_data["predictions"], key=lambda x: x["sample_idx"])
        y_true = np.array([s["y_true"][0] if isinstance(s["y_true"], list) else s["y_true"] for s in sorted_preds])
        
        # Calculate baseline accuracy
        baseline_acc = np.mean(orig_indices[:, 0] == y_true) * 100
        mapped_analysis[cl] = {}

        prefix = f"{cl}_{method}_{eval_strat}{retr_strat}"
        
        for k_key in k_dict.keys():
            if k_key == "per_class":
                continue
                
            match = re.search(r"\d+", k_key)
            if not match:
                continue
            v_num = match.group(0)
            
            tta_file = results_path / f"{prefix}nviews{v_num}.json"
            if not tta_file.exists():
                # Handle alternative filename patterns if necessary
                files = list(results_path.glob(f"{prefix}*{v_num}*.json"))
                if files:
                    tta_file = files[0]
                else:
                    continue

            tta_data = load_json(tta_file)
            later_indices, _ = get_top_k({
                "predictions": sorted(tta_data["predictions"], key=lambda x: x["sample_idx"])
            }, k=5)
            
            # Calculate TTA accuracies
            tta_top1_acc = np.mean(later_indices[:, 0] == y_true) * 100
            
            # Calculate Top-5 accuracy (is y_true anywhere in top 5)
            tta_top5_acc = np.mean([y in row for y, row in zip(y_true, later_indices)]) * 100
            
            # Combine with consistency rates
            mapped_analysis[cl][k_key] = {
                "baseline_top1_acc": round(baseline_acc, 2),
                "tta_top1_acc": round(tta_top1_acc, 2),
                "delta_acc": round(tta_top1_acc - baseline_acc, 2),
                "tta_top5_acc": round(tta_top5_acc, 2),
                "top1_agreement": k_dict[k_key]["top1"],
                "top5_any_order_agreement": k_dict[k_key]["top5_any_order"],
            }
            
    return mapped_analysis

def report_accuracy_agreement_mapping(mapped_analysis, method_name):
    print("\n" + "=" * 95)
    print(f"ACCURACY VS AGREEMENT CROSS-ANALYSIS - {method_name.upper()}")
    print("=" * 95)
    
    header = (
        f"{'Architecture':<32} {'Views':<6} "
        f"{'Base Acc':<10} {'TTA Acc':<10} {'Delta':<8} "
        f"{'Top1 Agree':<12} {'Top5 Overlap':<12}"
    )
    print(header)
    print("-" * len(header))
    
    for cl, k_dict in mapped_analysis.items():
        # Sort keys based on numeric view count
        sorted_keys = sorted(
            k_dict.keys(), 
            key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else x
        )
        
        for k_key in sorted_keys:
            metrics = k_dict[k_key]
            delta_str = f"{metrics['delta_acc']:+6.2f}%"
            
            print(
                f"{cl:<32} {k_key:<6} "
                f"{metrics['baseline_top1_acc']:>8.2f}% "
                f"{metrics['tta_top1_acc']:>8.2f}% "
                f"{delta_str:>7} "
                f"{metrics['top1_agreement']:>10.2f}% "
                f"{metrics['top5_any_order_agreement']:>10.2f}%"
            )
        print("-" * len(header))


if __name__ == "__main__":
    baseline_directory = "./results/baseline"
    results_position = ["geometric_tta", "adain_tta", "retristyle"]
    
    for rp in results_position:
        rp_path = rp if rp == "geometric_tta" else f"ablation/{rp}"
        results_directory = f"./results/{rp_path}/tta_inference/predictions/imagenet/test_r"
        
        if rp == "geometric_tta":
            settings = {"eval_strat": "vanilla", "method": "geometric", "retr_strat": ""}
        elif rp == "adain_tta":
            settings = {"eval_strat": "zero", "method": "adain_tta", "retr_strat": "_dino"}
        elif rp == "retristyle":
            settings = {"eval_strat": "vanilla", "method": "retristyle", "retr_strat": "_dino"}

        output_path = f"results/tta_agreement/tta_agreement_{settings['method']}_{settings['eval_strat']}{settings['retr_strat']}_summary.json" 

        # 1. Compute/load agreements and generate original metrics matrices
        results = analyze_and_report_agreement(
            baseline_dir=baseline_directory,
            results_dir=results_directory,
            best_settings=settings,
            output_filepath=output_path,
            dataset="imagenet",
            split="test_r",
            report=False
        )
        
        # If no files were found for a whole method pipeline, skip analysis steps safely
        if not results:
            continue

        # 2. Map consistency data with accuracy logs
        acc_agr_data = map_accuracy_with_agreement(
            agreement_json_path=output_path,
            baseline_dir=baseline_directory,
            results_dir=results_directory,
            best_settings=settings,
            dataset="imagenet",
            split="test_r"
        )
        
        # 3. Print the cross-analysis reporting matrix
        report_accuracy_agreement_mapping(acc_agr_data, method_name=settings['method'])

from pathlib import Path
from typing import Optional
import json
from tqdm import tqdm 
import numpy as np
from scipy.stats import chi2
from config.constants import ALL_CLASSIFIERS, TTA_STRATEGIES, DEFAULT_SEED, ALL_DATASETS, RETRIEVAL_STRATEGIES, ALL_SEEDS
from config.helpers import baseline_results

def get_prediction_filename(s_key, cfg, ds, cl, ev, retr, rfs, seed, sty=1, use_n=1):
    target_cfg = TTA_STRATEGIES.get(s_key, cfg)
    
    if s_key == "hybrid_tta":
        geo = (rfs - 1) - sty
        return "{ds}_{cl}_hybrid_geo{geo}_sty{sty}_{eval}_split{use_n}_nr{rfs}_seed{seed}_predictions.json".format(
            ds=ds,
            cl=cl,
            geo=f"{geo:02d}",
            sty=f"{sty:02d}",
            eval=ev,
            use_n=use_n,
            rfs=rfs,
            seed=seed,
        )

    fmt_kwargs = {
        "ds": ds,
        "cl": cl, 
        "eval": ev if ev else target_cfg.get("default_eval", "vanilla"), 
        "rfs": rfs,
        "seed": seed, 
        "retr": retr if retr is not None else target_cfg.get("default_retr", ""), 
    }              
    try:
        return target_cfg["template"].format(**fmt_kwargs)
    except KeyError:
        return target_cfg["template"].format(cl=cl, rfs=rfs, seed=seed)


def get_predictions(results: dict):
    """
    Extract true labels and predicted classes from a results JSON.

    Assumes:
        results["predictions"] = [
            {"y_true": ..., "y_pred": [...]},
            ...
        ]

    Returns:
        y_true: shape (N,)
        y_pred: shape (N,)
    """
    y_true = np.array(
        [sample["y_true"] for sample in results["predictions"]]
    )

    y_pred_matrix = np.array(
        [sample["y_pred"] for sample in results["predictions"]]
    )

    y_pred = np.argmax(y_pred_matrix, axis=1)

    return y_true, y_pred


def balanced_accuracy_score_np(y_true, y_pred, normal=False):
    """
    Balanced accuracy = mean recall over classes.
    """
    if normal:
        return float(np.mean(y_true == y_pred))
    
    classes = np.unique(y_true)

    recalls = []

    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) == 0:
            continue

        recall = np.mean(y_pred[mask] == cls)
        recalls.append(recall)

    return float(np.mean(recalls))


def stratified_bootstrap_difference(
    y_true,
    pred_baseline,
    pred_tta,
    normal=False,
    n_bootstrap=10_000,
    confidence_level=0.95,
    seed=42,
):
    """
    Stratified bootstrap CI for the difference in balanced accuracy:

        TTA balanced accuracy - baseline balanced accuracy

    Resampling is performed independently within each ground-truth class.
    """

    rng = np.random.default_rng(seed)

    classes = np.unique(y_true)

    # Store indices for every class.
    class_indices = {
        cls: np.where(y_true == cls)[0]
        for cls in classes
    }

    bootstrap_differences = np.empty(n_bootstrap)

    for b in tqdm(range(n_bootstrap)):

        sampled_indices = []

        # Stratified resampling:
        # sample the same number of examples from each class
        # as originally present in that class.
        for cls in classes:
            indices = class_indices[cls]

            sampled = rng.choice(
                indices,
                size=len(indices),
                replace=True,
            )

            sampled_indices.append(sampled)

        sampled_indices = np.concatenate(sampled_indices)

        y_true_b = y_true[sampled_indices]
        baseline_b = pred_baseline[sampled_indices]
        tta_b = pred_tta[sampled_indices]

        baseline_bal_acc = balanced_accuracy_score_np(
            y_true_b,
            baseline_b,
            normal
        )

        tta_bal_acc = balanced_accuracy_score_np(
            y_true_b,
            tta_b,
            normal
        )

        bootstrap_differences[b] = (
            tta_bal_acc - baseline_bal_acc
        )

    alpha = 1.0 - confidence_level

    lower = np.percentile(
        bootstrap_differences,
        100 * (alpha / 2),
    )

    upper = np.percentile(
        bootstrap_differences,
        100 * (1 - alpha / 2),
    )

    # Difference on the original test set
    baseline_bal_acc = balanced_accuracy_score_np(
        y_true,
        pred_baseline,
        normal
    )

    tta_bal_acc = balanced_accuracy_score_np(
        y_true,
        pred_tta,
        normal
    )

    observed_difference = tta_bal_acc - baseline_bal_acc

    return {
        "baseline_balanced_accuracy": baseline_bal_acc,
        "tta_balanced_accuracy": tta_bal_acc,
        "difference": observed_difference,
        "difference_percentage_points": observed_difference * 100,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_lower_percentage_points": float(lower * 100),
        "ci_upper_percentage_points": float(upper * 100),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def mcnemar_test(
    y_true,
    pred_baseline,
    pred_tta,
):
    """
    McNemar's test comparing paired baseline and TTA predictions.

    Uses the exact binomial test, which is preferable to the
    chi-square approximation when the number of discordant
    observations is small.
    """

    baseline_correct = pred_baseline == y_true
    tta_correct = pred_tta == y_true

    # Baseline correct, TTA wrong
    b = np.sum(
        baseline_correct & ~tta_correct
    )

    # Baseline wrong, TTA correct
    c = np.sum(
        ~baseline_correct & tta_correct
    )

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test:
        # Under H0, b and c have probability 0.5 each.
        from scipy.stats import binomtest

        p_value = binomtest(
            k=min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "baseline_correct_tta_wrong": int(b),
        "baseline_wrong_tta_correct": int(c),
        "discordant_pairs": int(discordant),
        "p_value": float(p_value),
    }


def calculate_statistical_comparison(
    baseline_path: Path,
    tta_path: Path,
    output_path: Path,
    normal=False,
    n_bootstrap=10_000,
    confidence_level=0.95,
    seed=42,
):
    """
    Calculate stratified bootstrap CI and McNemar's test
    for one baseline-vs-TTA comparison.

    Results are saved as JSON.
    """

    with open(baseline_path, "r") as f:
        baseline_results = json.load(f)

    with open(tta_path, "r") as f:
        tta_results = json.load(f)

    y_true_baseline, pred_baseline = get_predictions(
        baseline_results
    )

    y_true_tta, pred_tta = get_predictions(
        tta_results
    )

    if len(y_true_baseline) != len(y_true_tta):
        raise ValueError(
            f"Baseline {len(y_true_baseline)} and TTA {len(y_true_tta)} contain different numbers of predictions."
        )

    if not np.array_equal(y_true_baseline, y_true_tta):
        raise ValueError(
            "Ground-truth labels do not match between baseline and TTA."
        )

    y_true = y_true_baseline

    bootstrap_results = stratified_bootstrap_difference(
        y_true=y_true,
        pred_baseline=pred_baseline,
        pred_tta=pred_tta,
        normal=normal,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    mcnemar_results = mcnemar_test(
        y_true=y_true,
        pred_baseline=pred_baseline,
        pred_tta=pred_tta,
    )

    result = {
        "baseline_file": str(baseline_path),
        "tta_file": str(tta_path),
        "n_samples": int(len(y_true)),
        "n_classes": int(len(np.unique(y_true))),

        "bootstrap": bootstrap_results,

        "mcnemar": mcnemar_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(f"Saved statistical results to: {output_path}")

    return result

def set_against_path(against, cfg, ds, cl , seed, results_dir, split, s_key, rfs=1, sty=3, use_n=4):
    if against == "baseline":
        base_fname = get_prediction_filename("geometric_tta", cfg, ds, cl, "vanilla", "", rfs=1, seed=seed)
        # Baselines are stored at top-level split directory or without split subfolder depending on setup
        base_pred_path = results_dir / f"geometric_tta/tta_inference/predictions/{ds}/{split}" / base_fname
        
        if not base_pred_path.exists():
            base_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}" / base_fname
            if not base_pred_path.exists():
                return None
    else:
        eval = "zero" if against == "adain_tta" else "vanilla"
        retr = "" if against == "geometric_tta" else "dino"
        base_fname = get_prediction_filename(against, cfg, ds, cl, eval, retr, rfs, seed=seed, sty=sty, use_n=use_n)
        base_pred_path = results_dir / against / f"tta_inference/predictions/{ds}/{split}" / base_fname
    
    return base_pred_path

def run_all_statistical_comparisons(
    results_dir: Path,
    output_dir: Path,
    datasets: list,
    against: str = "baseline",
    normal: bool = False, 
    n_bootstrap: int = 10_000,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary_stats = {}
  
    for ds in datasets:
        all_summary_stats[ds] = {}
        split = "test"
        if ds == "eurosat":
            split = "ucmerced"
        elif ds == "imagenet":
            split = "test_r"

        for s_key, cfg in TTA_STRATEGIES.items():
            if s_key == against:
                continue
            ev = cfg.get("default_eval", "vanilla")
            retr = cfg.get("default_retr", "")
            all_summary_stats[ds][s_key] = {}
            for cl in ALL_CLASSIFIERS:
                all_summary_stats[ds][s_key][cl] = {}
                for seed in ALL_SEEDS:
                    
                    # 2. Loop through TTA step settings in axis
                    for rfs in cfg["axis"]:
                        if s_key == "hybrid_tta":
                            for sty_val in [1, 2, 3]:
                                for use_n in [1, sty_val + 1]:
                                    tta_fname = get_prediction_filename(
                                        s_key, cfg, ds, cl, ev, retr, rfs=rfs, seed=seed, sty=sty_val, use_n=use_n
                                    )
                                    tta_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}/{split}" / tta_fname

                                    base_pred_path = set_against_path(against, cfg, ds, cl, seed, results_dir, split, s_key,rfs, sty_val, use_n)
                                    if not base_pred_path.exists():
                                        continue
                                    if tta_pred_path.exists():
                                        out_json = output_dir / against / ds / s_key / cl / f"stats_nr{rfs}_sty{sty_val}_split{use_n}_seed{seed}.json"
                                        stats_res = calculate_statistical_comparison(
                                            baseline_path=base_pred_path,
                                            tta_path=tta_pred_path,
                                            output_path=out_json,
                                            normal=normal,
                                            n_bootstrap=n_bootstrap,
                                            seed=seed,
                                        )
                                        all_summary_stats[ds][s_key][cl][f"nr_{rfs}_sty_{sty_val}_split_{use_n}_seed_{seed}"] = stats_res
                        else:
                            tta_fname = get_prediction_filename(s_key, cfg, ds, cl, ev, retr, rfs=rfs, seed=seed)
                            tta_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}/{split}" / tta_fname
                            base_pred_path = set_against_path(against, cfg, ds, cl, seed, results_dir, split, s_key, rfs)
                            if not base_pred_path.exists():
                                continue                            
                            if tta_pred_path.exists():
                                out_json = output_dir / against / ds / s_key / cl / f"stats_rfs{rfs}_seed{seed}.json"
                                stats_res = calculate_statistical_comparison(
                                    baseline_path=base_pred_path,
                                    tta_path=tta_pred_path,
                                    output_path=out_json,
                                    normal=normal,
                                    n_bootstrap=n_bootstrap,
                                    seed=seed,
                                )
                                all_summary_stats[ds][s_key][cl][f"rfs_{rfs}_seed_{seed}"] = stats_res

    # Save aggregated execution metadata
    #with open(output_dir / "aggregated_summary.json", "w") as f:
    #    json.dump(all_summary_stats, f, indent=2)


import re

def run_list_comparisons(
    prediction_paths: str,
    results_dir: Path,
    output_dir: Path,
    dataset:str, 
    split: str, 
    n_bootstrap: int = 10_000,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary_stats = {}

    for cls in ALL_CLASSIFIERS:
        for retr in RETRIEVAL_STRATEGIES:
            tta_pred_path = preds.format(cl=cls, retr=retr)



            filename = Path(tta_pred_path)
            if not filename.exists():
                print(f"Skipping non-existent path: {tta_pred_path}")
                continue
            classifier = cls

            # Locate baseline matching dataset/split setup
            base_fname = f"{classifier}_vanilla.json"
            base_pred_path = Path(baseline_results(dataset=dataset, split=split, cls=cls, prediction=True))
           
            if not base_pred_path.exists():
                base_pred_path = results_dir / f"geometric_tta/tta_inference/predictions/{dataset}" / base_fname
                if not base_pred_path.exists():
                    print(f"Baseline not found for {tta_pred_path}, skipping.")
                    continue

            # Define structured output location
            stem = filename.stem
            out_json = output_dir / dataset / classifier / f"stats_{stem}.json"

            stats_res = calculate_statistical_comparison(
                baseline_path=base_pred_path,
                tta_path=filename,
                output_path=out_json,
                n_bootstrap=n_bootstrap,
            )

            all_summary_stats[stem] = stats_res

    with open(output_dir / "aggregated_summary.json", "w") as f:
        json.dump(all_summary_stats, f, indent=2)

if __name__ == "__main__":
    """
    preds = "/home/stud/nemmler/retristyle/results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/{cl}_adain_tta_zero_{retr}_nrefs32_seed71397589.json"
    run_list_comparisons(preds, 
                        Path("./results"),
                        Path("./results/statistics"),
                        dataset="imagenet", 
                        split="test_r")
    """
    for ag in ["baseline", "geometric_tta"]:
        run_all_statistical_comparisons(
            results_dir=Path("./results"),
            output_dir=Path("./results/statistics/acc"),
            datasets=["eurosat", "imagenet", "midog","camelyon17wilds", "epistr"],
            against=ag,
            normal=True
        )
   

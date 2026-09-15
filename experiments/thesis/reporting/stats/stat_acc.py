import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest
from tqdm import tqdm

from config.constants import (
    ALL_CLASSIFIERS,
    ALL_DATASETS,
    ALL_SEEDS,
    DEFAULT_SEED,
    RETRIEVAL_STRATEGIES,
    TTA_STRATEGIES,
)
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


_prediction_cache = {}


def get_predictions(results: dict):
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    y_pred_matrix = np.array([sample["y_pred"] for sample in results["predictions"]])
    y_pred = np.argmax(y_pred_matrix, axis=1)
    return y_true, y_pred


def load_predictions(path: Path):
    path_str = str(path)
    if path_str not in _prediction_cache:
        with open(path_str, "r") as f:
            results = json.load(f)
        _prediction_cache[path_str] = get_predictions(results)
    return _prediction_cache[path_str]


def balanced_accuracy_score_np(y_true, y_pred, normal=False):
    if normal:
        return float(np.mean(y_true == y_pred))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    n_classes = len(classes)

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[v] for v in y_true], dtype=np.int64)
    y_pred_idx = np.array([class_to_idx.get(v, -1) for v in y_pred], dtype=np.int64)

    totals = np.bincount(y_true_idx, minlength=n_classes)
    correct_mask = y_pred_idx == y_true_idx
    correct = np.bincount(y_true_idx[correct_mask], minlength=n_classes)

    recalls = np.divide(correct, totals, out=np.zeros(n_classes, dtype=np.float64), where=totals != 0)
    return float(np.mean(recalls))


def multi_seed_cluster_bootstrap_difference(
    seed_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    normal=False,
    n_bootstrap=10_000,
    confidence_level=0.95,
    seed=42,
):
    rng = np.random.default_rng(seed)
    seeds_list = list(seed_data.keys())
    n_seeds = len(seeds_list)

    preprocessed_seeds = {}
    for s_id in seeds_list:
        yt, pb, pt = seed_data[s_id]
        classes = np.unique(yt)
        class_indices = {cls: np.where(yt == cls)[0] for cls in classes}
        preprocessed_seeds[s_id] = {
            "y_true": yt,
            "pred_baseline": pb,
            "pred_tta": pt,
            "classes": classes,
            "class_indices": class_indices,
        }

    bootstrap_differences = np.empty(n_bootstrap)

    for b in tqdm(range(n_bootstrap), desc="Cluster Bootstrap"):
        sampled_seed_ids = rng.choice(seeds_list, size=n_seeds, replace=True)

        all_yt, all_pb, all_pt = [], [], []

        for s_id in sampled_seed_ids:
            s_meta = preprocessed_seeds[s_id]
            yt = s_meta["y_true"]
            pb = s_meta["pred_baseline"]
            pt = s_meta["pred_tta"]
            class_indices = s_meta["class_indices"]

            sampled_indices = []
            for cls in s_meta["classes"]:
                indices = class_indices[cls]
                sampled = rng.choice(indices, size=len(indices), replace=True)
                sampled_indices.append(sampled)

            sampled_indices = np.concatenate(sampled_indices)
            all_yt.append(yt[sampled_indices])
            all_pb.append(pb[sampled_indices])
            all_pt.append(pt[sampled_indices])

        pooled_yt = np.concatenate(all_yt)
        pooled_pb = np.concatenate(all_pb)
        pooled_pt = np.concatenate(all_pt)

        baseline_bal_acc = balanced_accuracy_score_np(pooled_yt, pooled_pb, normal)
        tta_bal_acc = balanced_accuracy_score_np(pooled_yt, pooled_pt, normal)
        bootstrap_differences[b] = tta_bal_acc - baseline_bal_acc

    alpha = 1.0 - confidence_level
    lower = np.percentile(bootstrap_differences, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_differences, 100 * (1 - alpha / 2))

    obs_base_accs, obs_tta_accs = [], []
    for s_id in seeds_list:
        yt, pb, pt = seed_data[s_id]
        obs_base_accs.append(balanced_accuracy_score_np(yt, pb, normal))
        obs_tta_accs.append(balanced_accuracy_score_np(yt, pt, normal))

    mean_baseline_acc = float(np.mean(obs_base_accs))
    mean_tta_acc = float(np.mean(obs_tta_accs))
    observed_difference = mean_tta_acc - mean_baseline_acc

    return {
        "baseline_balanced_accuracy_mean": mean_baseline_acc,
        "tta_balanced_accuracy_mean": mean_tta_acc,
        "difference": float(observed_difference),
        "difference_percentage_points": float(observed_difference * 100),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_lower_percentage_points": float(lower * 100),
        "ci_upper_percentage_points": float(upper * 100),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def mcnemar_test_pooled(seed_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    all_yt = np.concatenate([v[0] for v in seed_data.values()])
    all_pb = np.concatenate([v[1] for v in seed_data.values()])
    all_pt = np.concatenate([v[2] for v in seed_data.values()])

    baseline_correct = all_pb == all_yt
    tta_correct = all_pt == all_yt

    b = np.sum(baseline_correct & ~tta_correct)
    c = np.sum(~baseline_correct & tta_correct)
    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
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


def calculate_statistical_comparison_multi_seed(
    seed_paths: Dict[int, Tuple[Path, Path]],
    output_path: Path,
    normal=False,
    n_bootstrap=10_000,
    confidence_level=0.95,
    seed=42,
):
    seed_data = {}
    for s_id, (base_path, tta_path) in seed_paths.items():
        yt_base, pb = load_predictions(base_path)
        yt_tta, pt = load_predictions(tta_path)

        if len(yt_base) != len(yt_tta):
            raise ValueError(f"Seed {s_id}: baseline and TTA prediction sizes mismatch.")

        if not np.array_equal(yt_base, yt_tta):
            raise ValueError(f"Seed {s_id}: Ground-truth labels mismatch between baseline and TTA.")

        seed_data[s_id] = (yt_base, pb, pt)

    bootstrap_results = multi_seed_cluster_bootstrap_difference(
        seed_data=seed_data,
        normal=normal,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    mcnemar_results = mcnemar_test_pooled(seed_data=seed_data)

    first_s = list(seed_data.keys())[0]
    result = {
        "seeds": list(seed_paths.keys()),
        "baseline_files": {s: str(p[0]) for s, p in seed_paths.items()},
        "tta_files": {s: str(p[1]) for s, p in seed_paths.items()},
        "n_samples_per_seed": int(len(seed_data[first_s][0])),
        "n_classes": int(len(np.unique(seed_data[first_s][0]))),
        "bootstrap": bootstrap_results,
        "mcnemar": mcnemar_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved aggregated statistical results to: {output_path}")
    return result


def set_against_path(against, cfg, ds, cl, seed, results_dir, split, s_key, rfs=1, sty=3, use_n=4):
    if against == "baseline":
        base_fname = get_prediction_filename("geometric_tta", cfg, ds, cl, "vanilla", "", rfs=1, seed=seed)
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
        print(f"\n=== Starting dataset: {ds} ===")
        all_summary_stats[ds] = {}

        split = "test"
        if ds == "eurosat":
            split = "ucmerced"
        elif ds == "imagenet":
            split = "test_r"

        for s_key, cfg in TTA_STRATEGIES.items():
            if s_key == against:
                continue

            print(f"  Strategy: {s_key}")
            ev = cfg.get("default_eval", "vanilla")
            retr = cfg.get("default_retr", "")
            all_summary_stats[ds][s_key] = {}

            for cl in ALL_CLASSIFIERS:
                all_summary_stats[ds][s_key][cl] = {}

                for rfs in cfg["axis"]:
                    if s_key == "hybrid_tta":
                        for sty_val in [1, 2, 3]:
                            for use_n in [1, sty_val + 1]:
                                print(f"    Processing multi-seed: {ds} | {s_key} | {cl} | rfs={rfs} | sty={sty_val} | use_n={use_n}")

                                seed_paths = {}
                                missing_seed = False

                                for seed in ALL_SEEDS:
                                    tta_fname = get_prediction_filename(
                                        s_key, cfg, ds, cl, ev, retr,
                                        rfs=rfs, seed=seed,
                                        sty=sty_val, use_n=use_n,
                                    )
                                    tta_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}/{split}" / tta_fname
                                    base_pred_path = set_against_path(
                                        against, cfg, ds, cl, seed,
                                        results_dir, split, s_key,
                                        rfs, sty_val, use_n,
                                    )

                                    if base_pred_path is None or not base_pred_path.exists() or not tta_pred_path.exists():
                                        missing_seed = True
                                        break

                                    seed_paths[seed] = (base_pred_path, tta_pred_path)

                                if missing_seed or len(seed_paths) < len(ALL_SEEDS):
                                    print("      Missing required seed paths, skipping combination.")
                                    continue

                                out_json = output_dir / against / ds / s_key / cl / f"stats_nr{rfs}_sty{sty_val}_split{use_n}_all_seeds.json"
                                stats_res = calculate_statistical_comparison_multi_seed(
                                    seed_paths=seed_paths,
                                    output_path=out_json,
                                    normal=normal,
                                    n_bootstrap=n_bootstrap,
                                )
                                all_summary_stats[ds][s_key][cl][f"nr_{rfs}_sty_{sty_val}_split_{use_n}"] = stats_res

                    else:
                        print(f"    Processing multi-seed: {ds} | {s_key} | {cl} | rfs={rfs}")

                        seed_paths = {}
                        missing_seed = False

                        for seed in ALL_SEEDS:
                            tta_fname = get_prediction_filename(
                                s_key, cfg, ds, cl, ev, retr,
                                rfs=rfs, seed=seed,
                            )
                            tta_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}/{split}" / tta_fname
                            base_pred_path = set_against_path(
                                against, cfg, ds, cl, seed,
                                results_dir, split, s_key, rfs,
                            )

                            if base_pred_path is None or not base_pred_path.exists() or not tta_pred_path.exists():
                                missing_seed = True
                                break

                            seed_paths[seed] = (base_pred_path, tta_pred_path)

                        if missing_seed or len(seed_paths) < len(ALL_SEEDS):
                            print("      Missing required seed paths, skipping combination.")
                            continue

                        out_json = output_dir / against / ds / s_key / cl / f"stats_rfs{rfs}_all_seeds.json"
                        stats_res = calculate_statistical_comparison_multi_seed(
                            seed_paths=seed_paths,
                            output_path=out_json,
                            normal=normal,
                            n_bootstrap=n_bootstrap,
                        )
                        all_summary_stats[ds][s_key][cl][f"rfs_{rfs}"] = stats_res

        print(f"=== Finished dataset: {ds} ===\n")


if __name__ == "__main__":
    print("SCRIPT STARTED", flush=True)
    for acc_state in [True, False]:
        for ag in ["baseline", "geometric_tta"]:
            run_all_statistical_comparisons(
                results_dir=Path("./results"),
                output_dir=Path("./results/statistics/bal"),
                datasets=["imagenet", "midog", "camelyon17wilds", "epistr"],
                against=ag,
                normal=acc_state,
            )

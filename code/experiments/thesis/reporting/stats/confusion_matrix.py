import json
from pathlib import Path
import numpy as np
from code.config.constants import ALL_CLASSIFIERS, DEFAULT_SEED, TTA_STRATEGIES, ALL_DATASETS
from code.config.helpers import get_prediction_filename
from tqdm import tqdm
def get_predictions(results: dict):
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    y_pred_matrix = np.array(
        [sample["y_pred"] for sample in results["predictions"]]
    )
    y_pred = np.argmax(y_pred_matrix, axis=1)
    return y_true, y_pred


def categorize_samples(y_true, pred_base, pred_tta):
    base_correct = pred_base == y_true
    tta_correct = pred_tta == y_true

    # Sample categorization logic
    saved_mask = ~base_correct & tta_correct  # Made Correct
    stayed_pos_mask = base_correct & tta_correct  # Stayed Correct
    made_neg_mask = base_correct & ~tta_correct  # Made Incorrect
    stayed_neg_mask = ~base_correct & ~tta_correct  # Stayed Incorrect

    return {
        "counts": {
            "saved": int(np.sum(saved_mask)),
            "stayed_positive": int(np.sum(stayed_pos_mask)),
            "made_negative": int(np.sum(made_neg_mask)),
            "stayed_negative": int(np.sum(stayed_neg_mask)),
            "total_samples": int(len(y_true)),
        },
        "indices": {
            "saved": np.where(saved_mask)[0].tolist(),
            "stayed_positive": np.where(stayed_pos_mask)[0].tolist(),
            "made_negative": np.where(made_neg_mask)[0].tolist(),
            "stayed_negative": np.where(stayed_neg_mask)[0].tolist(),
        },
    }


def aggregate_transitions(
    results_dir: Path, output_dir: Path, ds: str
):

    aggregated_results = {}

    aggregated_results[ds] = {}
    split = "test"
    if ds == "eurosat":
        split = "ucmerced"
    elif ds == "imagenet":
        split = "test_r"

    for s_key, cfg in TTA_STRATEGIES.items():

        aggregated_results[ds][s_key] = {}
        ev = cfg.get("default_eval", "vanilla")
        retr = cfg.get("default_retr", "")

        for cl in ALL_CLASSIFIERS:
            aggregated_results[ds][s_key][cl] = {}
            seed = DEFAULT_SEED

            base_fname = f"{cl}_geometric_vanilla_nviews1_seed{DEFAULT_SEED}.json"
            base_pred_path = (
                results_dir
                / f"geometric_tta/tta_inference/predictions/{ds}/{split}"
                / base_fname
            )

            with open(base_pred_path, "r") as f:
                y_true_base, pred_base = get_predictions(json.load(f))

            output_path = output_dir / s_key / ds / cl 
            output_path.mkdir(parents=True, exist_ok=True)

            for rfs in cfg["axis"]:
                results = {}
                if s_key == "hybrid_tta":
                    
                    for sty_val in [1, 2, 3]:
                        for use_n in [1, sty_val + 1]:
                            tta_fname = get_prediction_filename(
                                s_key,
                                cfg,
                                ds,
                                cl,
                                ev,
                                retr,
                                rfs=rfs,
                                seed=seed,
                                sty=sty_val,
                                use_n=use_n,
                            )
                            tta_pred_path = (
                                results_dir
                                / s_key
                                / f"tta_inference/predictions/{ds}/{split}"
                                / tta_fname
                            )

                            if tta_pred_path.exists():
                                with open(tta_pred_path, "r") as f:
                                    _, pred_tta = get_predictions(
                                        json.load(f)
                                    )

                                key = f"nr_{rfs}_sty_{sty_val}_split_{use_n}_seed_{seed}"
                                results = categorize_samples(
                                    y_true_base, pred_base, pred_tta
                                )
                            else:
                                print(f"Nothing found for {tta_pred_path}")
                                continue
                            with open(output_path / f"{key}_sample_transitions_summary.json", "w") as f:
                                                json.dump(results, f)
                else:
                    tta_fname = get_prediction_filename(
                        s_key, cfg, ds, cl, ev, retr, rfs=rfs, seed=seed
                    )
                    tta_pred_path = (
                        results_dir
                        / s_key
                        / f"tta_inference/predictions/{ds}/{split}"
                        / tta_fname
                    )

                    if tta_pred_path.exists():
                        with open(tta_pred_path, "r") as f:
                            _, pred_tta = get_predictions(json.load(f))

                        key = f"rfs_{rfs}_seed_{seed}"
                        results = categorize_samples(
                            y_true_base, pred_base, pred_tta
                        )
                    else:
                        #print(f"Nothing found for {tta_pred_path}")
                        continue
                    with open(output_path / f"{key}_sample_transitions_summary.json", "w") as f:
                        json.dump(results, f)


if __name__ == "__main__":
    for ds in ["camelyon17wilds", "epistr", "imagenet","eurosat", "midog"]:
        aggregate_transitions(
            results_dir=Path("./results"),
            output_dir=Path("./results/sample_analysis"),
            ds=ds,
        )

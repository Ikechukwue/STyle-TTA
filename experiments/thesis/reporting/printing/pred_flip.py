import json
from pathlib import Path
import numpy as np

ALL_CLASSIFIERS = {
    "densenet121": "CNN",
    "swin_base_patch4_window7_224": "ViT",
    "dinov2_vitb14": "FM",
}

ALL_SEEDS = [71397589]

DATASET_SPLITS = {
    "imagenet": "test_r",
    "eurosat": "ucmerced",
    "midog": "test",
    "camelyon17wilds": "test",
    "epistr": "test",
}


def load_predictions(path: Path):
    with open(path, "r") as f:
        results = json.load(f)
    y_true = np.array([s["y_true"] for s in results["predictions"]])
    y_prob = np.array([s["y_pred"] for s in results["predictions"]])
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred


def get_baseline_predictions(base_dir: Path, ds: str, split: str, cl: str, seed: int):
    fname = f"{cl}_geometric_vanilla_nviews1_seed{seed}.json"
    path = base_dir / "geometric_tta" / "tta_inference" / "predictions" / ds / split / fname
    if not path.exists():
        return None, None
    return load_predictions(path)


def compute_method_flips(base_dir: Path, method: str, ds: str, split: str, cl: str, seed: int, param_val: int):
    y_true_base, y_pred_base = get_baseline_predictions(base_dir, ds, split, cl, seed)
    if y_true_base is None:
        return None
    
    if method == "retristyle":
        fname = f"{cl}_retristyle_vanilla_dino_nrefs{param_val}_seed{seed}.json"
        path = base_dir / "ablation" / "retristyle" / "tta_inference" / "predictions" / ds / split / fname
    elif method == "geometric":
        fname = f"{cl}_geometric_vanilla_nviews{param_val}_seed{seed}.json"
        path = base_dir / "geometric_tta" / "tta_inference" / "predictions" / ds / split / fname
    else:
        raise ValueError(f"Unknown method: {method}")

    if not path.exists():
        return None

    y_true_method, y_pred_method = load_predictions(path)

    base_correct = (y_pred_base == y_true_base)
    method_correct = (y_pred_method == y_true_method)

    c_to_c = np.mean(base_correct & method_correct) * 100
    c_to_w = np.mean(base_correct & ~method_correct) * 100
    w_to_c = np.mean(~base_correct & method_correct) * 100
    w_to_w = np.mean(~base_correct & ~method_correct) * 100

    return {
        "c2c": c_to_c,
        "c2w": c_to_w,
        "w2c": w_to_c,
        "w2w": w_to_w,
    }


def print_table(title: str, metrics_list: list):
    c2c = np.mean([m["c2c"] for m in metrics_list])
    c2w = np.mean([m["c2w"] for m in metrics_list])
    w2c = np.mean([m["w2c"] for m in metrics_list])
    w2w = np.mean([m["w2w"] for m in metrics_list])

    print(f"\n  --- {title} ---")
    print("  ┌─────────────────────────┬─────────────────────────┐")
    print(f"  │ Correct -> Correct (Keep)│ Correct -> Wrong (Harm) │")
    print(f"  │ {c2c:21.2f}% │ {c2w:21.2f}% │")
    print("  ├─────────────────────────┼─────────────────────────┤")
    print(f"  │  Wrong -> Correct (Fix) │  Wrong -> Wrong (Remain)│")
    print(f"  │ {w2c:21.2f}% │ {w2w:21.2f}% │")
    print("  └─────────────────────────┴─────────────────────────┘")
    print(f"    Net Gain (Fix - Harm): {w2c - c2w:+.2f}%")


def run_dual_analysis(results_dir: Path, refs: int = 16, views: int = 64):
    for ds, split in DATASET_SPLITS.items():
        print(f"\n================ DATASET: {ds.upper()} ================")
        for cl_raw, cl_display in ALL_CLASSIFIERS.items():
            print(f"\nModel: {cl_display} ({cl_raw})")

            retri_list, geo_list = [], []
            for seed in ALL_SEEDS:
                refs = 16 if ds == "imagenet" else 4
                m_retri = compute_method_flips(results_dir, "retristyle", ds, split, cl_raw, seed, refs)
                m_geo = compute_method_flips(results_dir, "geometric", ds, split, cl_raw, seed, views)

                if m_retri:
                    retri_list.append(m_retri)
                if m_geo:
                    geo_list.append(m_geo)

            if retri_list:
                print_table(f"RetriStyle (refs={refs})", retri_list)
            else:
                print(f"  [RetriStyle] Missing prediction files.")

            if geo_list:
                print_table(f"Geometric TTA (views={views})", geo_list)
            else:
                print(f"  [Geometric TTA] Missing prediction files.")


if __name__ == "__main__":
    results_base = Path("/home/stud/nemmler/retristyle/results")
    run_dual_analysis(results_base)

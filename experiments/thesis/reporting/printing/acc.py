from config.helpers import load_json, get_names, get_top_k, calc_top_k
from config.constants import ALL_CLASSIFIERS, TTA_STRATEGIES, DEFAULT_SEED
from pathlib import Path
import scipy.stats as stats
import numpy as np 

def print_acc_deltas():
    results = {}
    res_dir = Path("./results/geometric_tta/tta_inference/results/imagenet")
    template = TTA_STRATEGIES["geometric_tta"]["template"]
    datasets = ["val@test_r", "test_r"] # base, ood
    for cls in ALL_CLASSIFIERS:
        results[cls] = {}
        temp = template.format(cl=cls, rfs=1, seed=DEFAULT_SEED)
        base = load_json(res_dir / datasets[0] / temp)
        ood = load_json(res_dir / datasets[1] / temp) # Fixed index to datasets[1]

        base_acc = base["metrics"]["accuracy"]
        ood_acc = ood["metrics"]["accuracy"]
        delta = ood_acc - base_acc

        print(f"Classifier: {cls}")
        print(f"  Base Acc: {base_acc:.4f}")
        print(f"  OOD Acc:  {ood_acc:.4f}")
        print(f"  Delta:    {delta:+.4f}\n")

        results[cls] = {
            "base": base["metrics"],
            "ood": ood["metrics"]
        }

if __name__ == "__main__":
    print_acc_deltas()

import json
from config.helpers import load_json
from config.constants import ALL_CLASSIFIERS
from pathlib import Path
full = load_json("/home/stud/nemmler/retristyle/results/domain_gap/feature_space/test_r/cross_backbone_summary.json")
json_folder = Path("/home/stud/nemmler/retristyle/results/domain_gap/feature_space/test_r")

full_metrics = {"ood": full, 
                "baseline": {}}
for cls in ALL_CLASSIFIERS:
    json_name = f"{cls}_domain_gap.json"
    full_json = json_folder / json_name
    if not full_json.exists():
        print(f"Nothing found for {cls} at {full_json}")
        continue 
    
    side = load_json(full_json)["baseline_gap"]["global"]
    full_metrics["baseline"][cls] = side
gram_results = "/home/stud/nemmler/retristyle/results/domain_gap/feature_space/test_r/gram_summary.json"
gram_data = load_json(gram_results)

gram_base = gram_data["baseline_gap"]["global"]
gram_ood = gram_data["domain_gap"]["global"]
full_metrics["ood"]["gram_distance"] = gram_ood
full_metrics["baseline"]["gram_distance"]= gram_base


results_path = json_folder / "full_cross_domain_gap.json"
with open(results_path, 'w') as f:
    json.dump(full_metrics, f, indent=4)

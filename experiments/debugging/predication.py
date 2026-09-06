from pathlib import Path
from config.helpers import load_json

search_dirs = [
    Path("/home/stud/nemmler/retristyle/results/hybrid_tta/tta_inference/predictions/eurosat/ucmerced/"),
    Path("/home/stud/nemmler/retristyle/results/geometric_tta/tta_inference/predictions/eurosat/ucmerced/"),
    Path("/home/stud/nemmler/retristyle/results/ablation/retristyle/tta_inference/predictions/eurosat/ucmerced/")
]
for directory in search_dirs:
    for filepath in directory.glob("*.json"):
        data = load_json(str(filepath))
        length = len(data.get("predictions", []))
        if length != 600:
            print(f"FLAG: {filepath} length is {length} (expected 600)")

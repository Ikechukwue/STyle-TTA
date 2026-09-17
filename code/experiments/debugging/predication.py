from pathlib import Path
from code.config.helpers import load_json
from code.config.paths import OUTPUT_PATH

search_dirs = [
    OUTPUT_PATH / "hybrid_tta" / "tta_inference" / "predictions" / "eurosat" / "ucmerced",
    OUTPUT_PATH / "geometric_tta" / "tta_inference" / "predictions" / "eurosat" / "ucmerced",
    OUTPUT_PATH / "ablation" / "style_tta" / "tta_inference" / "predictions" / "eurosat" / "ucmerced",
]
for directory in search_dirs:
    for filepath in directory.glob("*.json"):
        data = load_json(str(filepath))
        length = len(data.get("predictions", []))
        if length != 600:
            print(f"FLAG: {filepath} length is {length} (expected 600)")

import json
from config.helpers import load_json
from experiments.data import (
    DATASET_SPLITS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    create_dataset,
)

test_set = create_dataset(
    dataset_name="eurosat", data_path="./data", split="ucmerced"
)
path = "/home/stud/nemmler/retristyle/data/augmented_cache/dino_eurosat_ucmerced_s71397589/manifest.json"
data = load_json(path)

completed_indices = list(range(len(test_set)))
new_data = {"completed_indices": completed_indices, "config": data["config"]}

with open(path, "w") as f:
    json.dump(new_data, f, indent=2)

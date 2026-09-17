import json
from code.config.helpers import load_json
from code.config.paths import DATA_PATH

path = DATA_PATH / "midog22_dataset" / "images" / "MIDOG2022_training_png.json"

data = load_json(path)

for k in data:
    print(k)
print(data["categories"][0])
# Inspect structure of a single image entry
print(data["images"][0].keys())

# Inspect structure of a single an  notation entry
print(data["annotations"][0].keys())


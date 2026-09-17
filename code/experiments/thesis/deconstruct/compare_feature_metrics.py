from code.config.helpers import get_names, load_json
import numpy as np
def print_feature_rankning(path:str):
    data = load_json(path)
    classifier = data["backbone"]
    target_doamin=data["splits"]["domain"]

    mmd = np.array([i for i in range(200)],[v["mmd"] for _, v in data["per_class"].items()])
    wasserstein = np.array([i for i in range(200)],[v["wasserstein"] for _, v in data["per_class"].items()])
    kl_symmetric = np.array([i for i in range(200)],[v["kl_symmetric"] for _, v in data["per_class"].items()])

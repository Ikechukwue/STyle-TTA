from .retri_info import create_retieval_json
from itertools import product
from code.config.constants import DEFAULT_SEED,  TRUE_SPLITS, ALL_SPLITS

if __name__ == "__main__":


    
    for m, s in ALL_SPLITS.items():
        if m in ["imagenet", "eurosat", "camelyon17wilds"]:
            continue
        output_path = f"./results/retrieval_mapping//{m}/retrieval_mapping_dino_test_r_s{DEFAULT_SEED}.json"
        create_retieval_json(ds=m, split=TRUE_SPLITS[s], output_path=output_path)

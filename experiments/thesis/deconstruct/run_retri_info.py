from .retri_info import create_retieval_json
from itertools import product
if __name__ == "__main__":

    methods = ["balanced_random"]
    seeds = [71397589, 133560673, 265017005]
    
    for m, s in product(methods, seeds):
        output_path = f"./results/retrieval_mapping/retrieval_mapping_{m}_test_r_s{s}.json"
        create_retieval_json(m, s, output_path)

import json
import numpy as np
import pandas as pd
import os 
from config.helpers import get_top_k, load_json, get_names
from config.constants import ALL_CLASSIFIERS
def results_to_summary_csv(results_path: str, output_dir:str):
    results = load_json(results_path)
    config = results["config"]
    
    results_name = os.path.basename(results_path)
    classifier = None
    for cl in ALL_CLASSIFIERS:
        if cl in results_name:
            classifier = cl
    title_parts = [config["dataset"], config["split"], classifier]
    #if config["tta_method"] != "geometric":
    #    title_parts.append(config["retrieval_strategy"])
    title_parts.extend([config["eval_strategy"]])
    
    filename = f"baseline_{'_'.join(title_parts)}.csv"
    
    top1_indices, top1_conf = get_top_k(results, k=1)
    top5_indices, top5_conf = get_top_k(results, k=5)
    
    top1_list = top1_indices.squeeze(axis=-1).tolist()
    top5_list = top5_indices.tolist()
    
    top1_conf_list = top1_conf.squeeze(axis=-1).tolist()
    top5_conf_list = top5_conf.tolist()
    rows = []
    for i, sample in enumerate(results["predictions"]):
        rows.append({
            "sample_idx": sample["sample_idx"],
            "y_true": sample["y_true"],
            "top1_idx": top1_list[i],
            "top1_conf": top1_conf_list[i],
            "top5_idx": top5_list[i],
            "top5_conf": top5_conf_list[i],
        })
        
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    
    #name_list = get_names()
    output_dir = "./results/csv_summary/baseline"
    for cl in ALL_CLASSIFIERS:
        results_path = f"./results/baseline/tta_inference/predictions/imagenet/test_r/{cl}_geometric_vanilla_nviews1_seed71397589.json"
        if os.path.exists(results_path):
            results_to_summary_csv(results_path, output_dir)
            print(f"Created summary for {cl}")
        else:
            print(f"Skipped {cl}")

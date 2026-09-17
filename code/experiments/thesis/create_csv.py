import json
import numpy as np
import pandas as pd
import os 
from code.config.helpers import get_top_k, load_json, get_names
from code.config.constants import ALL_CLASSIFIERS, TTA_STRATEGIES, ALL_SEEDS,N_REFS_VALUES, EVAL_STRATEGIES, RETRIEVAL_STRATEGIES
from pathlib import Path

def results_to_csv(results_path: str, output_dir: str):
    rows = []
    for cl in ALL_CLASSIFIERS:
        for tta in TTA_STRATEGIES:
            if tta == "hybrid_tta":
                continue
            for eval in EVAL_STRATEGIES:
                for retr in RETRIEVAL_STRATEGIES:
                    for seed in ALL_SEEDS:
                        for rfs in N_REFS_VALUES:
                            # 1. Fix template lookup
                            template = TTA_STRATEGIES[tta]["template"]
                            path = results_path.format(tta=tta, direction="results", template=template)
                            
                            # 2. Fix 'seed' parameter keyword argument
                            final_path = Path(path.format(
                                tta=tta, cl=cl, eval=eval, rfs=rfs, seed=seed, retr=retr if tta != "geometric_tta" else ""
                            ))
                            
                            if final_path.exists():
                                data = load_json(final_path)
                                rows.append({
                                    "tta strategy": tta if "/" not in tta else tta.split("/")[1],
                                    "classifier": cl, 
                                    "views": rfs, 
                                    "seed": seed, 
                                    "aggr": eval,
                                    "retr": retr if tta != "geometric_tta" else None,
                                    "acc": data["metrics"]["accuracy"],
                                    "bal_acc": data["metrics"]["balanced_accuracy"],
                                    "auc": data["metrics"]["auc"],
                                    "ece": data["metrics"]["ece"]
                                })
                    if tta == "geometric_tta":
                        break       
    # Convert gathered list of dicts to DataFrame and save to CSV
    df = pd.DataFrame(rows)
    output_path = Path(output_dir) / "results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def predictions_to_summary_csv(results_path: str, output_dir:str):
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
    output_dir = "./results/csv_summary"
    results_path = "./results/{tta}/tta_inference/{direction}/imagenet/test_r/{template}"
    results_to_csv(results_path, output_dir)

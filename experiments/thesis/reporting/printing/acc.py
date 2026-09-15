from config.helpers import load_json, get_classifier_name, get_baseline_results, get_names, get_top_k, calc_top_k
from config.constants import ALL_CLASSIFIERS, EVAL_STRATEGIES, RETRIEVAL_STRATEGIES, ALL_SEEDS, TTA_STRATEGIES, DEFAULT_SEED
from pathlib import Path
import scipy.stats as stats
import numpy as np 

def print_acc_deltas():
    results = {}
    res_dir = Path("./results/geometric_tta/tta_inference/results/imagenet")
    template = TTA_STRATEGIES["geometric_tta"]["template"]
    datasets = ["val@test_r", "test_r"] # base, ood
    for cls in ALL_CLASSIFIERS:
        results[cls] = {}
        temp = template.format(cl=cls, rfs=1, seed=DEFAULT_SEED)
        base = load_json(res_dir / datasets[0] / temp)
        ood = load_json(res_dir / datasets[1] / temp)

        base_acc = base["metrics"]["accuracy"]
        ood_acc = ood["metrics"]["accuracy"]
        delta = ood_acc - base_acc

        print(f"Classifier: {cls}")
        print(f"  Base Acc: {base_acc:.4f}")
        print(f"  OOD Acc:  {ood_acc:.4f}")
        print(f"  Delta:    {delta:+.4f}\n")

        results[cls] = {
            "base": base["metrics"],
            "ood": ood["metrics"]
        }

# Fixed filename generator that dynamically infers geo and accepts sty explicitly
def get_hybrid_filename(dataset, cl, eval_strategy, split, sty, nr, seed):
    geo = (nr - 1) - sty
    return f"{dataset}_{cl}_hybrid_geo{geo:02d}_sty{sty:02d}_{eval_strategy}_split{split}_nr{nr}_seed{seed}_results.json"

# Map showing which 'split' index corresponds to which 'sty' count in your files
STY_TO_SPLIT_MAP = {
    1: 1, # sty01 uses split1 (or split2)
    2: 1, # sty02 uses split1 (or split3)
    3: 1  # sty03 uses split1 (or split4)
}

def print_ablation_nrefs_multi(results_dir, strategy_keys, dataset, split):
    strategies = strategy_keys if strategy_keys else [cfg for cfg in TTA_STRATEGIES]
    for k in [1]:
        results_store = {}
        baselines = {}
        
        for s_key in strategies:
            cfg = TTA_STRATEGIES[s_key]
            results_store[s_key] = {}
            
            for cl in ALL_CLASSIFIERS:
                results_store[s_key][cl] = {}
                raw_data = {}
                
                for rfs in cfg["axis"]:
                    if s_key == "hybrid_tta":
                        nr_val = rfs
                        eval_strategy = cfg.get("default_eval", "vanilla")
                        
                        for sty_val in [1, 2, 3]:
                            for seed in ALL_SEEDS:
                                f1_name = get_hybrid_filename(
                                    dataset=dataset, cl=cl, eval_strategy=eval_strategy,
                                    split=1, sty=sty_val, nr=nr_val, seed=seed
                                )
                                f1 = results_dir / s_key / f"tta_inference/results/{dataset}" / f1_name
                                
                                split_multi = sty_val + 1
                                f2_name = get_hybrid_filename(
                                    dataset=dataset, cl=cl, eval_strategy=eval_strategy,
                                    split=split_multi, sty=sty_val, nr=nr_val, seed=seed
                                )
                                f2 = results_dir / s_key / f"tta_inference/results/{dataset}" / f2_name

                                for mode, path in [("orig_only", f1), ("multi_source", f2)]:
                                    if path.exists():
                                        data = load_json(path)
                                        metrics = data.get("metrics", {})
                                        
                                        key_str = f"{rfs}_sty{sty_val}_{mode}"
                                        if key_str not in raw_data:
                                            raw_data[key_str] = {
                                                "accs": [], "bal_accs": [], "aucs": [], "eces": []
                                            }
                                        
                                        m_dict = raw_data[key_str]
                                        m_dict["accs"].append(metrics.get("accuracy", 0) * 100 if k == 1 else metrics.get("top5_accuracy", 0) * 100)
                                        m_dict["bal_accs"].append(metrics.get("balanced_accuracy", 0) * 100)
                                        m_dict["aucs"].append(metrics.get("auc", 0))
                                        m_dict["eces"].append(metrics.get("ece", 0))
                                        
                        for key_str, vals in raw_data.items():
                            if vals["accs"]:
                                results_store[s_key][cl][key_str] = {
                                    "acc_mean": np.mean(vals["accs"]), "acc_std": np.std(vals["accs"]),
                                    "bal_acc_mean": np.mean(vals["bal_accs"]), "bal_acc_std": np.std(vals["bal_accs"]),
                                    "auc_mean": np.mean(vals["aucs"]), "auc_std": np.std(vals["aucs"]),
                                    "ece_mean": np.mean(vals["eces"]), "ece_std": np.std(vals["eces"]),
                                }
                    else:
                        accs, bal_accs, aucs, eces = [], [], [], []
                        for seed in ALL_SEEDS:
                            if s_key == "ablation/adain_tta":
                                f_name = cfg["template"].format(cl=cl, eval="zero", retr="dino", rfs=rfs, seed=seed)
                            elif s_key == "ablation/retristyle":
                                f_name = cfg["template"].format(cl=cl, eval="vanilla", retr="dino", rfs=rfs, seed=seed)
                            elif s_key == "geometric_tta":
                                f_name = cfg["template"].format(cl=cl, eval="vanilla", rfs=rfs, seed=seed, retr="")
                            else:
                                f_name = cfg["template"].format(cl=cl, rfs=rfs, seed=seed)
                        
                            f = results_dir / s_key / f"tta_inference/results/{dataset}/{split}" / f_name
                            if not f.exists(): 
                                continue
                            
                            data = load_json(f)
                            metrics = data.get("metrics", {})
                            accs.append(metrics.get("accuracy", 0) * 100 if k == 1 else calc_top_k(results_dir / s_key / f"tta_inference/predictions/{dataset}/{split}" / f_name, k))
                            bal_accs.append(metrics.get("balanced_accuracy", 0) * 100)
                            aucs.append(metrics.get("auc", 0))
                            eces.append(metrics.get("ece", 0))
                        
                        if accs:
                            results_store[s_key][cl][str(rfs)] = {
                                "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                                "bal_acc_mean": np.mean(bal_accs), "bal_acc_std": np.std(bal_accs),
                                "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
                                "ece_mean": np.mean(eces), "ece_std": np.std(eces),
                            }

        print(f"\n=== Top-{k} Accuracy & Metrics Statistics (Dataset: {dataset} | Split: {split}) ===")
        
        for s_key in strategies:
            label = TTA_STRATEGIES[s_key]["label"]
            print(f"\nStrategy: {label} ({s_key})")
            
            for cl in ALL_CLASSIFIERS:
                cl_name = get_classifier_name(cl)[0]
                print(f"  Classifier: {cl_name}")
                
                axis_vals = TTA_STRATEGIES[s_key]["axis"]
                for rfs in axis_vals:
                    if s_key == "hybrid_tta":
                        for sty_val in [1, 2, 3]:
                            key_orig = f"{rfs}_sty{sty_val}_orig_only"
                            m_orig = results_store[s_key].get(cl, {}).get(key_orig, None)
                            if m_orig:
                                print(
                                    f"    nr={rfs:<2} | sty={sty_val:02d} | [Geo Source: 1 Orig Image] (split1) -> "
                                    f"Top-{k} Acc: {m_orig['acc_mean']:.2f} ± {m_orig['acc_std']:.2f} | "
                                    f"Bal Acc: {m_orig['bal_acc_mean']:.2f} ± {m_orig['bal_acc_std']:.2f}"
                                )
                            
                            key_multi = f"{rfs}_sty{sty_val}_multi_source"
                            m_multi = results_store[s_key].get(cl, {}).get(key_multi, None)
                            if m_multi:
                                geo_sources = 1 + sty_val
                                split_used = sty_val + 1
                                print(
                                    f"    nr={rfs:<2} | sty={sty_val:02d} | [Geo Sources: {geo_sources} (1 Orig + {sty_val} Style)] (split{split_used}) -> "
                                    f"Top-{k} Acc: {m_multi['acc_mean']:.2f} ± {m_multi['acc_std']:.2f} | "
                                    f"Bal Acc: {m_multi['bal_acc_mean']:.2f} ± {m_multi['bal_acc_std']:.2f}"
                                )
                    else:
                        m = results_store[s_key].get(cl, {}).get(str(rfs), None)
                        if m:
                            print(
                                f"    n_refs={str(rfs):<10} -> "
                                f"Top-{k} Acc: {m['acc_mean']:.2f} ± {m['acc_std']:.2f} | "
                                f"Bal Acc: {m['bal_acc_mean']:.2f} ± {m['bal_acc_std']:.2f}"
                            )

def print_dynamic_results(
    results_dir,
    dataset,
    split,
    strategy_keys=None,
    cls=None,
    evals="default",
    retrs="default",
    rfss="default",
    seeds="default",
    k=1
):
    strategies = [strategy_keys] if isinstance(strategy_keys, str) else strategy_keys
    if not strategies:
        strategies = list(TTA_STRATEGIES.keys())

    classifiers = [cls] if isinstance(cls, str) else (cls if cls else ALL_CLASSIFIERS)

    for s_key in strategies:
        cfg = TTA_STRATEGIES.get(s_key, {})
        label = cfg.get("label", s_key)
        
        eval_list = [cfg.get("default_eval", "vanilla")] if evals == "default" else ([evals] if isinstance(evals, str) else (evals if evals is not None else EVAL_STRATEGIES))
        retr_list = [cfg.get("default_retr", "dino")] if retrs == "default" else ([retrs] if isinstance(retrs, str) else (retrs if retrs is not None else RETRIEVAL_STRATEGIES))
        rfs_list = cfg.get("axis", [1]) if rfss == "default" else ([rfss] if isinstance(rfss, (int, tuple)) else rfss)
        seed_list = [DEFAULT_SEED] if seeds == "default" else ([seeds] if isinstance(seeds, int) else seeds)

        print(f"\n=== Strategy: {label} ({s_key}) | Dataset: {dataset} | Split: {split} ===")

        for cl in classifiers:
            cl_name = get_classifier_name(cl)[0]
            print(f"\nClassifier: {cl_name}")

            for ev in eval_list:
                for retr in retr_list:
                    for rfs in [1]:#rfs_list:
                        accs, bal_accs, aucs, eces = [], [], [], []

                        for seed in seed_list:
                            if s_key == "hybrid_tta" and isinstance(rfs, tuple):
                                n_views, n_refs = rfs
                                f_name = get_hybrid_filename(
                                    dataset=dataset,
                                    cl=cl,
                                    eval_strategy=ev,
                                    split=n_refs,
                                    nr=n_views,
                                    seed=seed
                                )
                                f = results_dir / s_key / f"tta_inference/results/{dataset}" / f_name
                            else:
                                if s_key=="geometric_tta":
                                    retr = ""
                                fmt_kwargs = {"cl": cl, "rfs": rfs, "seed": seed, "eval": ev, "retr": retr}
                                try:
                                    f_name = cfg["template"].format(**fmt_kwargs)
                                except KeyError:
                                    f_name = cfg["template"].format(cl=cl, rfs=rfs, seed=seed)
                                f = results_dir / s_key / f"tta_inference/results/{dataset}/{split}" / f_name

                            if not f.exists():

                                continue

                            data = load_json(f)
                            metrics = data.get("metrics", {})

                            acc = metrics.get("accuracy", 0) * 100 if k == 1 else metrics.get("top5_accuracy", 0) * 100
                            accs.append(acc)
                            bal_accs.append(metrics.get("balanced_accuracy", 0) * 100)
                            aucs.append(metrics.get("auc", 0))
                            eces.append(metrics.get("ece", 0))

                        if accs:
                            print(
                                f"  [eval={ev:<7} | retr={retr:<5} | n_refs={str(rfs):<8}] -> "
                                f"Top-{k} Acc: {np.mean(accs):.2f}% | "
                                f"Bal Acc: {np.mean(bal_accs):.2f}% | "
                                f"AUC: {np.mean(aucs):.4f} | "
                                f"ECE: {np.mean(eces):.4f}"
                            )

if __name__ == "__main__":
    for ds in ["eurosat", "midog", "camelyon17wilds", "epistr", "imagenet"]:
        split = "test" 
        if ds == "eurosat": 
            split = "ucmerced"
        elif ds == "imagenet":
            split = "test_r"
        print_dynamic_results(Path("./results"), ds, split, "geometric_tta")
        #print_ablation_nrefs_multi(Path("./results"), ["geometric_tta"], ds, split)

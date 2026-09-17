import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from code.config.constants import ALL_CLASSIFIERS, ALL_SEEDS, TTA_STRATEGIES, ALL_DATASETS
from code.config.helpers import get_classifier_name, load_json

def get_hybrid_filename(dataset, cl, eval_strategy, split, sty, nr, seed):
    geo = (nr - 1) - sty
    return f"{dataset}_{cl}_hybrid_geo{geo:02d}_sty{sty:02d}_{eval_strategy}_split{split}_nr{nr}_seed{seed}_results.json"

def get_geometric_filename(cl, eval_strategy, nr, seed):
    cfg_geo = TTA_STRATEGIES.get("geometric_tta", {})
    template = cfg_geo.get("template", "{cl}_geometric_{eval}_nviews{rfs}_seed{seed}.json")
    return template.format(cl=cl, eval=eval_strategy, rfs=nr, seed=seed, retr="")

def plot_hybrid_tta_performance(results_dir, dataset, target_split, classifiers=None, metric="accuracy", save_path=None):
    cls_list = classifiers if classifiers else ALL_CLASSIFIERS
    cfg_hybrid = TTA_STRATEGIES.get("hybrid_tta", {})
    eval_strategy = cfg_hybrid.get("default_eval", "vanilla")
    axis_nrs = cfg_hybrid.get("axis", [4, 8, 16, 32, 64])
    
    records = []
    found_count = 0
    missing_count = 0

    print(f"Starting data collection for dataset: '{dataset}' (split: '{target_split}')...")
    
    for cl in cls_list:
        cl_display = get_classifier_name(cl)[0]
        print(f"\nProcessing classifier: {cl_display} ({cl})")
        
        for nr_val in axis_nrs:
            for seed in ALL_SEEDS:
                # Load Pure Geometric Baseline
                geo_fname = get_geometric_filename(cl, eval_strategy, nr_val, seed)
                geo_path = Path(results_dir) / "geometric_tta" / "tta_inference" / "results" / dataset / target_split / geo_fname
                
                if geo_path.exists():
                    data = load_json(geo_path)
                    val = data.get("metrics", {}).get(metric, 0)
                    if metric in ["accuracy", "top5_accuracy", "balanced_accuracy"]:
                        val *= 100
                    
                    records.append({
                        "classifier": cl_display,
                        "nr": nr_val,
                        "sty": "pure_geo",
                        "source_mode": "Pure Geometric",
                        "value": val,
                        "seed": seed
                    })
                    found_count += 1
                else:
                    missing_count += 1
                    if missing_count <= 3:
                        print(f"  [MISSING GEO] {geo_path}")

                # Load Hybrid TTA Variations
                for sty_val in [1, 2, 3]:
                    splits_to_check = [
                        (1, "orig_only"),
                        (sty_val + 1, "multi_source")
                    ]
                    
                    for split_val, source_mode in splits_to_check:
                        f_name = get_hybrid_filename(
                            dataset=dataset,
                            cl=cl,
                            eval_strategy=eval_strategy,
                            split=split_val,
                            sty=sty_val,
                            nr=nr_val,
                            seed=seed
                        )
                        f_path = Path(results_dir) / "hybrid_tta" / "tta_inference" / "results" / dataset / f_name
                        
                        if not f_path.exists():
                            missing_count += 1
                            if missing_count <= 3:
                                print(f"  [MISSING HYBRID] {f_path}")
                            continue
                            
                        data = load_json(f_path)
                        val = data.get("metrics", {}).get(metric, 0)
                        if metric in ["accuracy", "top5_accuracy", "balanced_accuracy"]:
                            val *= 100

                        records.append({
                            "classifier": cl_display,
                            "nr": nr_val,
                            "sty": f"sty={sty_val:02d}",
                            "source_mode": source_mode,
                            "value": val,
                            "seed": seed
                        })
                        found_count += 1

    print(f"\nData collection complete.")
    print(f"Total entries loaded: {found_count}")
    print(f"Total paths missing: {missing_count}")

    if not records:
        print("ERROR: No valid result files were found.")
        return

    df = pd.DataFrame(records)
    print(f"DataFrame created with {len(df)} rows. Generating plot...")

    # Color mapping: pure_geo is black, sty lines keep distinct colors
    palette = {
        "pure_geo": "black",
        "sty=01": "#1f77b4",
        "sty=02": "#ff7f0e",
        "sty=03": "#2ca02c"
    }

    # Style mapping: orig_only and pure_geo are solid, multi_source is dotted (1 pt on, 1 pt off)
    dashes = {
        "Pure Geometric": "",
        "orig_only": "",
        "multi_source": (1, 1)
    }

    g = sns.FacetGrid(
        df, 
        col="classifier", 
        col_wrap=min(len(cls_list), 3), 
        height=4, 
        aspect=1.2, 
        sharey=False
    )
    
    g.map_dataframe(
        sns.lineplot,
        x="nr",
        y="value",
        hue="sty",
        style="source_mode",
        palette=palette,
        dashes=dashes,
        markers=True,
        err_style="band"
    )

    g.set_axis_labels("Total Views (NR)", metric.replace("_", " ").title())
    g.set_titles(col_template="{col_name}")
    
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(axis_nrs)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

    g.add_legend(title="Configurations")
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle(f"Hybrid vs Pure Geometric TTA Metrics (Dataset: {dataset})")

    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    for ds in ALL_DATASETS.keys():
        if ds == "imagenet":
            split = "test_r"
        elif ds == "eurosat":
            split = "ucmerced"
        else:
            split = "test"
        plot_hybrid_tta_performance(
            results_dir="./results",
            dataset=ds,
            target_split=split,
            metric="accuracy",
            save_path=f"figures/hybrid/{ds}_hybrid_vs_geo_tta.png"
        )

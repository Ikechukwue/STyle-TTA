from pathlib import Path

import numpy as np

from config.helpers import load_json, get_classifier_name
from config.constants import (
    ALL_CLASSIFIERS,
    ALL_DATASETS,
    TRUE_SPLITS,
    ALL_SPLITS,
    ALL_SEEDS,
    DEFAULT_SEED,
    TTA_STRATEGIES,
)

def latex_escape(text):
        """Escape characters that have special meaning in LaTeX."""
        text = str(text)

        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

def generate_hybrid_summary_table(
    results_dir,
    dataset,
    split,
    views=(8, 16, 32, 64),
    splits_to_compare=(1, 4),
    fixed_sty=3,
    caption=None,
    label=None,
    output_path=None,
    include_std=False,
):
    results_dir = Path(results_dir)

    if caption is None:
        caption = f"Hybrid TTA and Geometric baseline performance comparison on {ALL_DATASETS[dataset]}."

    if label is None:
        label = f"tab:{ALL_DATASETS[dataset]}_hybrid_summary"

    def get_hybrid_filename(dataset, cl, eval_strategy, split_val, sty, nr, seed):
        geo = (nr - 1) - sty
        return (
            f"{dataset}_{cl}_hybrid_"
            f"geo{geo:02d}_"
            f"sty{sty:02d}_"
            f"{eval_strategy}_"
            f"split{split_val}_"
            f"nr{nr}_"
            f"seed{seed}_"
            f"results.json"
        )

    def get_geo_filename(dataset, cl, eval_strategy, nr, seed):
        return (
            f"{cl}_geometric_vanilla_"
            f"nviews{nr}_"
            f"seed{seed}.json"
        )

    def extract_acc(m_data):
        if "balanced_accuracy" in m_data:
            return m_data["balanced_accuracy"] * 100
        elif "accuracy" in m_data:
            return m_data["accuracy"] * 100
        return None

    def load_hybrid_metrics(cl, nr, target_split):
        values = []
        cfg = TTA_STRATEGIES.get("hybrid_tta", {})
        eval_strategy = cfg.get("default_eval", "vanilla")

        for seed in ALL_SEEDS:
            f_name = get_hybrid_filename(
                dataset=dataset,
                cl=cl,
                eval_strategy=eval_strategy,
                split_val=target_split,
                sty=fixed_sty,
                nr=nr,
                seed=seed,
            )
            f = results_dir / "hybrid_tta" / "tta_inference" / "results" / dataset / f_name

            if not f.exists():
                continue

            data = load_json(f)
            val = extract_acc(data.get("metrics", {}))
            if val is not None:
                values.append(val)

        if not values:
            return {"mean": np.nan, "std": np.nan}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    def load_geo_metrics(cl, nr):
        values = []
        cfg = TTA_STRATEGIES.get("geometric_tta", {})
        eval_strategy = cfg.get("default_eval", "vanilla")

        for seed in ALL_SEEDS:
            f_name = get_geo_filename(
                dataset=dataset,
                cl=cl,
                eval_strategy=eval_strategy,
                nr=nr,
                seed=seed,
            )
            # Try specific directory, fallback to base geometric_tta dir
            f = results_dir / "geometric_tta" / "tta_inference" / "results" / dataset / TRUE_SPLITS[split] / f_name 
            print(f)
            if not f.exists():
                f = results_dir / "geometric_tta" / f_name

            if not f.exists():
                continue

            data = load_json(f)
            val = extract_acc(data.get("metrics", {}))
            if val is not None:
                values.append(val)

        if not values:
            return {"mean": np.nan, "std": np.nan}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    data = {}
    for cl in ALL_CLASSIFIERS:
        data[cl] = {"geo": {}}
        for nr in views:
            data[cl]["geo"][nr] = load_geo_metrics(cl=cl, nr=nr)
            data[cl][nr] = {}
            for sp in splits_to_compare:
                data[cl][nr][sp] = load_hybrid_metrics(cl=cl, nr=nr, target_split=sp)

    def format_val(mean, std, decimals=2):
        if np.isnan(mean):
            return "--"
        if not include_std or np.isnan(std) or float(f"{std:.{decimals}f}") == 0:
            return f"{mean:.{decimals}f}"
        return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll" + "c" * len(views) + "}",
        r"\toprule",
        r" & & \multicolumn{" + str(len(views)) + r"}{c}{\textbf{Total Ensemble Size}} \\",
        rf"\cmidrule(lr){{3-{2 + len(views)}}}",
        r"\textbf{Backbone} & \textbf{Configuration} & " + " & ".join([f"\\textbf{{{v}}}" for v in views]) + r" \\",
        r"\midrule",
    ]

    configs = [
        ("geo", r"Geo-TTA only"),
        (1, r"Hybrid S1 (3 style + geo@orig)"),
        (4, r"Hybrid S2 (3 style + geo distributed)"),
    ]

    for cl_idx, cl in enumerate(ALL_CLASSIFIERS):
        cl_name = latex_escape(get_classifier_name(cl)[0])
        num_configs = len(configs)

        best_means = {}
        for nr in views:
            means = []
            for cfg_key, _ in configs:
                if cfg_key == "geo":
                    means.append(data[cl]["geo"][nr]["mean"])
                else:
                    means.append(data[cl][nr][cfg_key]["mean"])
            valid_means = [m for m in means if not np.isnan(m)]
            best_means[nr] = max(valid_means) if valid_means else np.nan

        for cfg_idx, (cfg_key, label_text) in enumerate(configs):
            backbone_cell = f"\\multirow{{{num_configs}}}{{*}}{{{cl_name}}}" if cfg_idx == 0 else ""
            row = [backbone_cell, label_text]

            for nr in views:
                if cfg_key == "geo":
                    mean = data[cl]["geo"][nr]["mean"]
                    std = data[cl]["geo"][nr]["std"]
                else:
                    mean = data[cl][nr][cfg_key]["mean"]
                    std = data[cl][nr][cfg_key]["std"]

                val_str = format_val(mean, std)

                if not np.isnan(mean) and np.isclose(mean, best_means[nr]):
                    val_str = f"\\textbf{{{val_str}}}"

                row.append(val_str)

            lines.append(" & ".join(row) + r" \\")

        if cl_idx < len(ALL_CLASSIFIERS) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])

    latex_code = "\n".join(lines)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(latex_code, encoding="utf-8")

    return latex_code

if __name__ == "__main__":
    RESULTS_DIR = Path("results")
    VIEWS = (8, 16, 32, 64)
    SPLITS_TO_COMPARE = (1, 4)
    FIXED_STY = 3

    for dataset, split in ALL_SPLITS.items():
        generate_hybrid_summary_table(
            results_dir=RESULTS_DIR,
            dataset=dataset,
            split=split,
            views=VIEWS,
            splits_to_compare=SPLITS_TO_COMPARE,
            fixed_sty=FIXED_STY,
            caption=(
                f"Hybrid TTA balanced accuracy on "
                f"{ALL_DATASETS[dataset]} "
                f"(split 1 vs split 4, sty={FIXED_STY})."
            ),
            label=(
                f"tab:{ALL_DATASETS[dataset]}_"
                f"hybrid_split1_vs_4"
            ),
            output_path=(
                RESULTS_DIR
                / "latex_tables"
                / f"{dataset}_hybrid_split1_vs_4.tex"
            ),
            include_std=True,
        )

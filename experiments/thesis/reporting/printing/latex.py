from pathlib import Path

import numpy as np

from config.helpers import load_json, get_classifier_name
from config.constants import (
    ALL_CLASSIFIERS,
    ALL_DATASETS,
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

def generate_latex_table(
    results_dir,
    dataset,
    split,
    strategy_keys,
    rfs_values,
    table_mode="balanced",
    caption=None,
    label=None,
    output_path=None,
    include_std=True,
):
    """
    Generate a LaTeX comparison table from TTA result files.

    Parameters
    ----------
    results_dir : Path
        Root results directory, e.g. Path("./results").

    dataset : str
        Dataset name, e.g. "eurosat".

    split : str
        Dataset split, e.g. "ucmerced".

    strategy_keys : list[str]
        Strategies to compare. Use "baseline" for the baseline.

    rfs_values : dict
        Maps each strategy to the desired n_refs value.

        Example:
            {
                "ablation/retristyle": 4,
                "geometric_tta": 64,
            }

        The baseline does not require an rfs value.

    table_mode : str
        "balanced" -> one row per classifier, balanced accuracy only.

        "comprehensive" -> three rows per classifier:
            Balanced Accuracy
            AUC
            ECE

    caption : str or None
        LaTeX table caption.

    label : str or None
        LaTeX table label.

    output_path : Path or str or None
        If provided, writes the LaTeX table to this file.

    include_std : bool
        If True, print mean ± std.
    """

    results_dir = Path(results_dir)

    if label is None:
        label = f"tab:{ALL_DATASETS[dataset]}_{mode}"

    if caption is None:
            if table_mode == "balanced":
                caption = f"Balanced Acc(\\%) {ALL_DATASETS[dataset]} $\\rightarrow$ {ALL_SPLITS[dataset]} results"
            else:
                caption = f"Full Comprehensive {ALL_DATASETS[dataset]} $\\rightarrow$ {ALL_SPLITS[dataset]} results"
    if table_mode not in {"balanced", "comprehensive"}:
        raise ValueError(
            "table_mode must be 'balanced' or 'comprehensive'"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    
    def fill_template(key,cl, rfs, seed):
        s_key = "geometric_tta" if key == "baseline" else key
        cfg = TTA_STRATEGIES[s_key]
        if s_key == "hybrid_tta":
            real_rfs, geo_fac = rfs
            f_name = cfg["template"].format(cl=cl, geo=geo_fac, rfs=real_rfs, seed=seed)
        elif s_key == "ablation/adain_tta":
            f_name = cfg["template"].format(cl=cl, eval="zero", retr="dino", rfs=rfs, seed=seed)
        elif s_key == "ablation/retristyle":
            f_name = cfg["template"].format(cl=cl, eval="vanilla", retr="dino", rfs=rfs, seed=seed)
        elif s_key == "geometric_tta":
            f_name = cfg["template"].format(cl=cl, eval="vanilla", rfs=rfs, seed=seed, retr="")
        else:
            f_name = cfg["template"].format(cl=cl, rfs=rfs, seed=seed)
        return f_name      
    def latex_classifier_name(cl):
        name = get_classifier_name(cl)[0]

        # Preserve names containing LaTeX-sensitive characters.
        return latex_escape(name)

    def format_value(mean, std, decimals=2):
        if np.isnan(mean):
            return "--"

        if not include_std or np.isnan(std):
            return f"{mean:.{decimals}f}"
        formatted_std = f"{std:.{decimals}f}"
        suffix = "" if float(formatted_std) == 0 else f" $\\pm$ {formatted_std}"
        return f"{mean:.{decimals}f}{suffix}"

    def get_result_path(strategy_key, filename):
        return (
            Path("./results/")
            / strategy_key
            / "tta_inference"
            / "results"
            / dataset
            / split
            / filename
        )

    # ------------------------------------------------------------------
    # Load metrics for one classifier / strategy / n_refs
    # ------------------------------------------------------------------

    def load_strategy_metrics(strategy_key, cl, rfs=None):
        values = {
            "acc":[],
            "bal_acc": [],
            "auc": [],
            "ece": [],
        }

        # --------------------------------------------------------------
        # Baseline
        # --------------------------------------------------------------
        if strategy_key == "baseline":
            #
            # Adjust this path/template if your baseline files have a
            # different naming convention.
            #
            # This assumes get_baseline_results() is available in
            # config.helpers and returns a result dictionary.
            #
            from config.helpers import get_baseline_results

            try:
                baseline_path = Path("./results/geometric_tta/tta_inference/results") / dataset / split / f"{cl}_geometric_vanilla_nviews1_seed{DEFAULT_SEED}.json"
                print(baseline_path)
                baseline = load_json(baseline_path)
            except TypeError:
                # Fallback for projects where get_baseline_results has
                # a different signature.
                baseline = get_baseline_results(
                    results_dir,
                    dataset,
                    split,
                    cl,
                )

            if baseline is None:
                return {
                    metric: {"mean": np.nan, "std": np.nan}
                    for metric in values
                }

            # Support either {"metrics": {...}} or directly {...}.
            metrics = baseline.get("metrics", baseline)

            if "balanced_accuracy" in metrics:
                values["bal_acc"].append(
                    metrics["balanced_accuracy"] * 100
                )
            if "accuracy" in metrics:
                values["acc"].append(
                    metrics["accuracy"] * 100
                )
            if "auc" in metrics:
                values["auc"].append(metrics["auc"])

            if "ece" in metrics:
                values["ece"].append(metrics["ece"])

        # --------------------------------------------------------------
        # TTA strategies
        # --------------------------------------------------------------
        else:
            if strategy_key not in TTA_STRATEGIES:
                raise KeyError(
                    f"Unknown strategy '{strategy_key}'. "
                    f"Available strategies: "
                    f"{list(TTA_STRATEGIES.keys())}"
                )

            if rfs is None:
                raise ValueError(
                    f"No rfs value provided for strategy "
                    f"'{strategy_key}'."
                )

            cfg = TTA_STRATEGIES[strategy_key]

            for seed in ALL_SEEDS:

                if strategy_key == "hybrid_tta":
                    real_rfs, geo_fac = rfs

                    f_name = cfg["template"].format(
                        cl=cl,
                        geo=geo_fac,
                        rfs=real_rfs,
                        seed=seed,
                    )

                elif strategy_key == "ablation/adain_tta":
                    f_name = cfg["template"].format(
                        cl=cl,
                        eval="zero",
                        retr="dino",
                        rfs=rfs,
                        seed=seed,
                    )

                elif strategy_key == "ablation/retristyle":
                    f_name = cfg["template"].format(
                        cl=cl,
                        eval="vanilla",
                        retr="dino",
                        rfs=rfs,
                        seed=seed,
                    )

                elif strategy_key == "geometric_tta":
                    f_name = cfg["template"].format(
                        cl=cl,
                        eval="vanilla",
                        rfs=rfs,
                        seed=seed,
                        retr="",
                    )

                else:
                    f_name = cfg["template"].format(
                        cl=cl,
                        rfs=rfs,
                        seed=seed,
                    )

                f = get_result_path(
                    strategy_key,
                    f_name,
                )
                print(f)
                if not f.exists():
                    continue

                data = load_json(f)
                metrics = data.get("metrics", {})

                if "balanced_accuracy" in metrics:
                    values["bal_acc"].append(
                        metrics["balanced_accuracy"] * 100
                    )

                if "auc" in metrics:
                    values["auc"].append(metrics["auc"])

                if "ece" in metrics:
                    values["ece"].append(metrics["ece"])

        # --------------------------------------------------------------
        # Aggregate
        # --------------------------------------------------------------
        result = {}

        for metric, vals in values.items():

            if not vals:
                result[metric] = {
                    "mean": np.nan,
                    "std": np.nan,
                }
                continue

            result[metric] = {
                "mean": float(np.mean(vals)),
                "std": (
                    float(np.std(vals, ddof=1))
                    if len(vals) > 1
                    else 0.0
                ),
            }

        return result

    # ------------------------------------------------------------------
    # Load all results
    # ------------------------------------------------------------------

    data = {}

    for cl in ALL_CLASSIFIERS:
        data[cl] = {}

        for strategy_key in strategy_keys:
            rfs = rfs_values.get(strategy_key)

            data[cl][strategy_key] = load_strategy_metrics(
                strategy_key=strategy_key,
                cl=cl,
                rfs=rfs,
            )

    # ------------------------------------------------------------------
    # Strategy labels
    # ------------------------------------------------------------------

    strategy_labels = {}

    for strategy_key in strategy_keys:

        if strategy_key == "baseline":
            strategy_labels[strategy_key] = "Baseline"

        elif strategy_key in TTA_STRATEGIES:
            strategy_labels[strategy_key] = TTA_STRATEGIES[
                strategy_key
            ]["label"]

        else:
            strategy_labels[strategy_key] = strategy_key

    # ------------------------------------------------------------------
    # Start LaTeX
    # ------------------------------------------------------------------

    lines = []

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    # ==================================================================
    # BALANCED ACCURACY
    # ==================================================================

    if table_mode == "balanced":

        lines.append(
            r"\begin{tabular}{l"
            + "c" * len(strategy_keys)
            + "}"
        )

        lines.append(r"\toprule")

        header = [r"\textbf{Backbone}"]

        for strategy_key in strategy_keys:
            views = "" if strategy_key == "baseline" else f"($N={rfs_values[strategy_key]}$)"
            header.append(
                r"\textbf{"
                + strategy_labels[strategy_key]
                + views
                + "}"
            )

        lines.append(
            " & ".join(header) + r" \\"
        )

        lines.append(r"\midrule")

        for cl in ALL_CLASSIFIERS:

            means = [
                data[cl][strategy_key]["bal_acc"]["mean"]
                for strategy_key in strategy_keys
            ]

            valid_means = [
                x for x in means
                if not np.isnan(x)
            ]

            best = (
                max(valid_means)
                if valid_means
                else np.nan
            )

            row = [latex_classifier_name(cl)]

            for strategy_key in strategy_keys:

                mean = data[cl][strategy_key]["bal_acc"]["mean"]
                std = data[cl][strategy_key]["bal_acc"]["std"]

                value = format_value(
                    mean,
                    std,
                    decimals=2,
                )

                if (
                    not np.isnan(mean)
                    and not np.isnan(best)
                    and np.isclose(mean, best)
                ):
                    value = r"\textbf{" + value + "}"

                row.append(value)

            lines.append(
                " & ".join(row) + r" \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
    # ==================================================================
    # COMPREHENSIVE
    # ==================================================================

    elif table_mode == "comprehensive":

        lines.append(r"\small")

        lines.append(
            r"\begin{tabular}{ll"
            + "c" * len(strategy_keys)
            + "}"
        )

        lines.append(r"\toprule")

        header = [
            r"\textbf{Classifier}",
            r"\textbf{Metric}",
        ]

        for strategy_key in strategy_keys:
            views = "" if strategy_key == "baseline" else f"($N={rfs_values[strategy_key]}$)"
            header.append(
                r"\textbf{"
                + strategy_labels[strategy_key]
                + views
                + "}"
            )

        lines.append(
            " & ".join(header) + r" \\"
        )

        lines.append(r"\midrule")

        metrics = [
            (
                "bal_acc",
                r"Bal Acc (\%) $\uparrow$",
                2,
                True,
            ),
            (
                "auc",
                r"AUC $\uparrow$",
                4,
                True,
            ),
            (
                "ece",
                r"ECE $\downarrow$",
                4,
                False,
            ),
        ]

        for cl_idx, cl in enumerate(ALL_CLASSIFIERS):

            cl_name = latex_classifier_name(cl)

            for metric_idx, (
                metric,
                metric_label,
                decimals,
                higher_is_better,
            ) in enumerate(metrics):

                means = [
                    data[cl][strategy_key][metric]["mean"]
                    for strategy_key in strategy_keys
                ]

                valid_means = [
                    x for x in means
                    if not np.isnan(x)
                ]

                if valid_means:
                    best = (
                        max(valid_means)
                        if higher_is_better
                        else min(valid_means)
                    )
                else:
                    best = np.nan

                if metric_idx == 0:
                    classifier_cell = (
                        f"\\multirow{{3}}{{*}}{{{cl_name}}}"
                    )
                else:
                    classifier_cell = ""

                row = [
                    classifier_cell,
                    metric_label,
                ]

                for strategy_key in strategy_keys:

                    mean = data[cl][strategy_key][metric]["mean"]
                    std = data[cl][strategy_key][metric]["std"]

                    value = format_value(
                        mean,
                        std,
                        decimals=decimals,
                    )

                    if (
                        not np.isnan(mean)
                        and not np.isnan(best)
                        and np.isclose(mean, best)
                    ):
                        value = r"\textbf{" + value + "}"

                    row.append(value)

                lines.append(
                    " & ".join(row) + r" \\"
                )

            if cl_idx < len(ALL_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    # ------------------------------------------------------------------
    # Print / save
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Generated LaTeX table")
    print("=" * 80)
    print(latex)
    print("=" * 80)

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            latex,
            encoding="utf-8",
        )

        print(
            f"\nLaTeX table saved to: {output_path}"
        )

    return latex


def generate_reference_count_table(
    results_dir,
    dataset,
    split,
    retristyle_refs,
    geometric_refs,
    caption=None,
    label=None,
    output_path=None,
    include_std=True,
):
    """
    Generate a LaTeX table showing balanced accuracy across
    different reference counts.

    Produces columns such as:

        Backbone | ReTriStyle 2 | ReTriStyle 4 |
                 | Geometric 2 | Geometric 4 | ...

    Only balanced accuracy is reported.
    """

    results_dir = Path(results_dir)

    if caption is None:
        caption = (
            f"Balanced accuracy across reference counts on "
            f"{ALL_DATASETS[dataset]} ({ALL_SPLITS[dataset]})."
        )

    if label is None:
        label = f"tab:{ALL_DATASETS[dataset]}_full_refs"

    # Build strategy list where each entry represents one
    # strategy/reference-count combination.
    columns = []

    for rfs in retristyle_refs:
        columns.append(
            ("ablation/retristyle", rfs, f"ReTriStyle ({rfs})")
        )

    for rfs in geometric_refs:
        columns.append(
            ("geometric_tta", rfs, f"Geometric ({rfs})")
        )

    # ------------------------------------------------------------------
    # Reuse the single-value table loader by loading one result at
    # a time. This keeps the filename handling in one place.
    # ------------------------------------------------------------------

    def load_metrics(strategy_key, cl, rfs):

        values = []

        cfg = TTA_STRATEGIES[strategy_key]

        for seed in ALL_SEEDS:

            if strategy_key == "ablation/retristyle":
                f_name = cfg["template"].format(
                    cl=cl,
                    eval="vanilla",
                    retr="dino",
                    rfs=rfs,
                    seed=seed,
                )

            elif strategy_key == "geometric_tta":
                f_name = cfg["template"].format(
                    cl=cl,
                    eval="vanilla",
                    rfs=rfs,
                    seed=seed,
                    retr="",
                )

            else:
                raise ValueError(
                    f"Unsupported reference-count strategy: "
                    f"{strategy_key}"
                )

            f = (
                results_dir
                / strategy_key
                / "tta_inference"
                / "results"
                / dataset
                / split
                / f_name
            )

            if not f.exists():
                continue

            data = load_json(f)
            metrics = data.get("metrics", {})

            if "balanced_accuracy" in metrics:
                values.append(
                    metrics["balanced_accuracy"] * 100
                )

        if not values:
            return np.nan, np.nan

        mean = float(np.mean(values))

        std = (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0
        )

        return mean, std

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    data = {}

    for cl in ALL_CLASSIFIERS:
        data[cl] = {}

        for strategy_key, rfs, label_text in columns:
            data[cl][(strategy_key, rfs)] = load_metrics(
                strategy_key,
                cl,
                rfs,
            )

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        (
            r"\begin{tabular}{l"
            + "c" * len(columns)
            + "}"
        ),
        r"\toprule",
    ]

    header = [r"\textbf{Backbone}"]

    for _, _, label_text in columns:
        header.append(
            r"\textbf{" + label_text + "}"
        )

    lines.append(
        " & ".join(header) + r" \\"
    )

    lines.append(r"\midrule")

    for cl in ALL_CLASSIFIERS:

        means = [
            data[cl][(strategy_key, rfs)][0]
            for strategy_key, rfs, _ in columns
        ]

        valid = [
            x for x in means
            if not np.isnan(x)
        ]

        best = max(valid) if valid else np.nan

        row = [latex_escape(get_classifier_name(cl)[0])]

        for strategy_key, rfs, _ in columns:

            mean, std = data[cl][(strategy_key, rfs)]

            if np.isnan(mean):
                value = "--"
            elif include_std:
                value = (
                    f"{mean:.2f} $\\pm$ {std:.2f}"
                )
            else:
                value = f"{mean:.2f}"

            if (
                not np.isnan(mean)
                and not np.isnan(best)
                and np.isclose(mean, best)
            ):
                value = r"\textbf{" + value + "}"

            row.append(value)

        lines.append(
            " & ".join(row) + r" \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    print("\n" + "=" * 80)
    print("Generated LaTeX reference-count table")
    print("=" * 80)
    print(latex)
    print("=" * 80)

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            latex,
            encoding="utf-8",
        )

        print(
            f"\nLaTeX table saved to: {output_path}"
        )

    return latex


if __name__ == "__main__":

    RESULTS_DIR = Path("./results")
    # ================================================================
    # Datasets
    # ================================================================

    DATASETS = [
        ("midog", "test"),
        ("eurosat", "ucmerced"),
        ("camelyon17wilds", "test"),
        ("epistr", "test"),
        ("imagenet", "test_r"),
    ]

    # ================================================================
    # Reference counts
    # ================================================================

    DEFAULT_RETRISTYLE_REFS = [2, 4]
    DEFAULT_GEOMETRIC_REFS = [2, 4, 8, 16, 32, 64]

    IMAGENET_RETRISTYLE_REFS = [2, 4, 8, 16]
    IMAGENET_GEOMETRIC_REFS = [2, 4, 8, 16, 32, 64]

    # ================================================================
    # Main comparison reference counts
    # ================================================================

    DEFAULT_MAIN_RETRISTYLE = 4
    DEFAULT_MAIN_GEOMETRIC = 64

    IMAGENET_MAIN_RETRISTYLE = 16
    IMAGENET_MAIN_GEOMETRIC = 64

    # ================================================================
    # Generate tables
    # ================================================================

    for dataset, split in DATASETS:

        print("\n")
        print("=" * 100)
        print(
            f"DATASET: {dataset} | SPLIT: {split}"
        )
        print("=" * 100)

        # ------------------------------------------------------------
        # Dataset-specific reference counts
        # ------------------------------------------------------------

        if dataset == "imagenet":

            retristyle_refs = IMAGENET_RETRISTYLE_REFS
            geometric_refs = IMAGENET_GEOMETRIC_REFS

            main_retristyle = IMAGENET_MAIN_RETRISTYLE
            main_geometric = IMAGENET_MAIN_GEOMETRIC

        else:

            retristyle_refs = DEFAULT_RETRISTYLE_REFS
            geometric_refs = DEFAULT_GEOMETRIC_REFS

            main_retristyle = DEFAULT_MAIN_RETRISTYLE
            main_geometric = DEFAULT_MAIN_GEOMETRIC

        # ------------------------------------------------------------
        # 1. Compact balanced-accuracy table
        #
        # Baseline | ReTriStyle | Geometric
        # ------------------------------------------------------------

        print(
            "\nGenerating compact Balanced Accuracy table..."
        )
        mode = "balanced"
        generate_latex_table(
            results_dir=RESULTS_DIR,
            dataset=dataset,
            split=split,
            strategy_keys=[
                "baseline",
                "ablation/retristyle",
                "geometric_tta",
            ],
            rfs_values={
                "ablation/retristyle": main_retristyle,
                "geometric_tta": main_geometric,
            },
            table_mode=mode,
            caption=None,
            label=None,
            output_path=(
                RESULTS_DIR
                / "latex_tables"
                / f"{dataset}_{mode}.tex"
            ),
            include_std=True,
        )

        # ------------------------------------------------------------
        # 2. Full reference-count table
        # ------------------------------------------------------------

        print(
            "\nGenerating full reference-count table..."
        )

        generate_reference_count_table(
            results_dir=RESULTS_DIR,
            dataset=dataset,
            split=split,
            retristyle_refs=retristyle_refs,
            geometric_refs=geometric_refs,
            caption=(
                f"Balanced accuracy across reference counts on "
                f"{ALL_DATASETS[dataset]}."
            ),
            label=f"tab:{ALL_DATASETS[dataset]}_full_refs",
            output_path=(
                RESULTS_DIR
                / "latex_tables"
                / f"{dataset}_full_refs.tex"
            ),
            include_std=True,
        )

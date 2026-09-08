from pathlib import Path
import numpy as np
from config.helpers import load_json, get_classifier_name, latex_escape, baseline_results
from config.constants import (
    ALL_CLASSIFIERS,
    ALL_DATASETS,
    ALL_SPLITS,
    ALL_SEEDS,
    DEFAULT_SEED,
    TTA_STRATEGIES,
    TRUE_SPLITS,
    CLEAN_CLASSIFIERS
)

import re

TARGET_CLASSIFIERS = [
    "densenet121",
    "swin_base_patch4_window7_224",
    "dinov2_vitb14",
    
]

CLASSIFIER_SHORT_NAMES = {
    "densenet121": "CNN",
    "dinov2_vitb14": "ViT",
    "swin_base_patch4_window7_224": "FM",
}
def short_classifier_name(cl):
    """Return mapped short name if present, otherwise fallback to default helper."""

    return CLASSIFIER_SHORT_NAMES[cl]

def latex_escape(text):
    """Escape characters that have special meaning in LaTeX."""
    text = str(text)

    # Clean non-breaking spaces (U+00A0) that break LaTeX compilation
    text = text.replace("\u00a0", " ")

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


def build_benchmarks_table(
    table_mode,
    ALL_CLASSIFIERS,
    ALL_SPLITS,
    strategy_keys,
    data,
    caption,
    label,
    latex_classifier_name,
    TRUE_SPLITS,
    strategy_label,
    format_pvalue,
):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    if table_mode == "benchmarks":
        lines.append(r"\begin{tabular}{llcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Backbone} & \textbf{Benchmark} & \textbf{Baseline (\%)} & "
            r"\textbf{Strategy} & \textbf{TTA (\%$\uparrow$)} & "
            r"\textbf{$\Delta$ (95\% CI)} & \textbf{$p$} & "
            r"\textbf{ECE $\downarrow$} & \textbf{AUC $\uparrow$} \\"
        )
        lines.append(r"\midrule")

        for cl_idx, cl in enumerate(TARGET_CLASSIFIERS):
            cl_name = latex_escape(latex_classifier_name(cl))

            valid_accs = [
                data[cl][ds][sk]["tta_acc"]
                for ds in ALL_SPLITS
                for sk in strategy_keys
                if data[cl].get(ds, {}).get(sk) is not None
                and "tta_acc" in data[cl][ds][sk]
            ]
            max_acc = max(valid_accs) if valid_accs else None

            printable_rows = []
            for ds, ds_split in ALL_SPLITS.items():
                for strategy_key in strategy_keys:
                    if strategy_key == "ablation/adain_tta":
                        continue
                    if data[cl].get(ds, {}).get(strategy_key) is not None:
                        printable_rows.append((ds, ds_split, strategy_key))

            total_cl_rows = len(printable_rows)
            cl_rendered = False

            for ds_idx, (ds, ds_split) in enumerate(ALL_SPLITS.items()):
                ds_rows = [r_tuple for r_tuple in printable_rows if r_tuple[0] == ds]
                ds_row_count = len(ds_rows)
                ds_rendered = False

                for r_ds, r_split, strategy_key in ds_rows:
                    r = data[cl][r_ds][strategy_key]

                    auc = (
                        f"{r['metrics'].get('auc', float('nan')):.2f}"
                        if r.get("metrics")
                        else "--"
                    )
                    ece = (
                        f"{r['metrics'].get('ece', float('nan')):.2f}"
                        if r.get("metrics")
                        else "--"
                    )

                    classifier_cell = (
                        f"\\multirow{{{total_cl_rows}}}{{*}}{{{cl_name}}}"
                        if not cl_rendered
                        else ""
                    )
                    cl_rendered = True

                    clean_split = latex_escape(TRUE_SPLITS[r_split])
                    dataset_cell = (
                        f"\\multirow{{{ds_row_count}}}{{*}}{{{clean_split}}}"
                        if not ds_rendered
                        else ""
                    )

                    baseline_cell = (
                        f"\\multirow{{{ds_row_count}}}{{*}}{{{r['baseline_acc']:.2f}}}"
                        if not ds_rendered
                        else ""
                    )
                    ds_rendered = True

                    delta_prefix = "+" if r["delta_pp"] >= 0 else ""
                    delta_ci = f"{delta_prefix}{r['delta_pp']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"

                    acc_str = f"{r['tta_acc']:.2f}"
                    if max_acc is not None and r["tta_acc"] == max_acc:
                        acc_str = f"\\textbf{{{acc_str}}}"

                    clean_strat = latex_escape(strategy_label(strategy_key))

                    row = [
                        classifier_cell,
                        dataset_cell,
                        baseline_cell,
                        clean_strat,
                        acc_str,
                        delta_ci,
                        format_pvalue(r["p_value"]),
                        ece,
                        auc,
                    ]
                    lines.append(" & ".join(row) + r" \\")

            if cl_idx < len(ALL_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")

    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def format_pvalue(p, threshold=0.001):
    """Format a p-value for display, handling float-underflow zeros."""
    if p is None or np.isnan(p):
        return "--"
    if p < threshold:
        return f"$<${threshold}"
    return f"{p:.3f}"


def get_metrics(path:str):
    results_path = path.replace("predictions", "results")

    if Path(results_path).exists(): 
        res_data = load_json(results_path)
    else:
        print(results_path)
        return {}
    return res_data["metrics"]

def load_transition_stats(results_dir, dataset, strategy_key, cl, rfs, seed=None, against="baseline",split= 1 , sty=1):
    """
    Load bootstrap + McNemar transition statistics for one
    (dataset, strategy, classifier, rfs) combination.

    Expects files at:
        {results_dir}/statistics/{dataset}/{strategy_key}/{cl}/
            stats_rfs{rfs}_seed{seed}.json

    Returns a dict with baseline/tta accuracy, the CI, and the
    saved/corrupted counts, or None if the file doesn't exist.
    """

    if seed is None:
        seed = DEFAULT_SEED
    if strategy_key == "hybrid_tta":
        str_file_key = f"stats_nr{rfs}_sty{sty}_split{split}_seed{seed}.json"
    else:
        str_file_key = f"stats_rfs{rfs}_seed{seed}.json"

    if strategy_key == against:
        f = (
            Path(results_dir)
            / "statistics"
            / "baseline"
            / dataset
            / strategy_key
            / cl
            / str_file_key
        )

        if not f.exists():
            print(f)
            return None

        data = load_json(f)
        metrics = get_metrics(data["tta_file"])
        bs = data.get("bootstrap", {})

        return {
            "n_samples": None,
            "baseline_acc": bs.get("tta_balanced_accuracy", np.nan) * 100,
            "tta_acc": bs.get("tta_balanced_accuracy", np.nan) * 100,
            "delta_pp": None,
            "ci_lower": None,
            "ci_upper": None,
            "saved": None,
            "corrupted": None,
            "saved_pct": None,
            "corrupted_pct": None,
            "rescue_ratio": None,
            "p_value": None,
            "metrics": metrics,
        }
    
    else: 

        f = (
            Path(results_dir)
            / "statistics"
            / against
            / dataset
            / strategy_key
            / cl
            / str_file_key
        )

        if not f.exists():
            print(f)
            return None

        data = load_json(f)
        bs = data.get("bootstrap", {})
        mc = data.get("mcnemar", {})

        saved = mc.get("baseline_wrong_tta_correct")
        corrupted = mc.get("baseline_correct_tta_wrong")
        n_samples = data.get("n_samples")
        metrics = get_metrics(data["tta_file"])

        return {
            "n_samples": n_samples,
            "baseline_acc": bs.get("baseline_balanced_accuracy", np.nan) * 100,
            "tta_acc": bs.get("tta_balanced_accuracy", np.nan) * 100,
            "delta_pp": bs.get("difference_percentage_points", np.nan),
            "ci_lower": bs.get("ci_lower_percentage_points", np.nan),
            "ci_upper": bs.get("ci_upper_percentage_points", np.nan),
            "saved": saved,
            "corrupted": corrupted,
            "saved_pct": (saved / n_samples * 100) if saved is not None and n_samples else np.nan,
            "corrupted_pct": (corrupted / n_samples * 100) if corrupted is not None and n_samples else np.nan,
            "rescue_ratio": (saved / corrupted) if corrupted else np.nan,
            "p_value": mc.get("p_value", np.nan),
            "metrics": metrics,
        }

def get_baseline_results(cl, dataset, split):
    base_path = baseline_results(cl, dataset, split)
    data = load_json(base_path)
    return data["metrics"]

def generate_transition_table(
    results_dir,
    dataset,
    split,
    strategy_keys,
    rfs_values,
    seed=None,
    table_mode="compact",
    caption=None,
    label=None,
    output_dir=None,
    against="baseline",
    hybrid_views=(8, 16, 32, 64),
    hybrid_splits=(1, 4),
    hybrid_sty = 3
):
    """
    Generate a LaTeX saved/corrupted (McNemar transition) table
    from precomputed bootstrap + McNemar statistics files.

    Parameters
    ----------
    strategy_keys : list[str]
        TTA strategies to report transitions for (e.g.
        ["ablation/retristyle"] for a single-strategy "compact"
        table, or several for a "comprehensive" side-by-side table).
        Does not include "baseline" -- the baseline is embedded in
        each stats file already.

    rfs_values : dict
        Maps each strategy to the n_refs value to load, same
        convention as generate_latex_table.

    table_mode : str
        "compact"       -> one row per classifier, single strategy,
                            flat columns (Baseline/TTA acc, Delta,
                            Saved, Corrupted, Rescue ratio, p).
        "comprehensive" -> one multirow block per classifier, one
                            sub-row per strategy in strategy_keys.
    """

    results_dir = Path(results_dir)

    if seed is None:
        seed = DEFAULT_SEED



    if label is None:
        label = f"tab:{ALL_DATASETS[dataset]}_transitions"

    if caption is None:
        caption = (
            f"Sample-level transitions (Baseline $\\rightarrow$ TTA), "
            f"{ALL_DATASETS[dataset]} $\\rightarrow$ {ALL_SPLITS[dataset]}."
        )

    def latex_classifier_name(cl):
        return latex_escape(get_classifier_name(cl)[1])

    def strategy_label(strategy_key):
        return TTA_STRATEGIES[strategy_key]["label"]

    # ------------------------------------------------------------------
    # Load all results
    # ------------------------------------------------------------------

    data = {}
    if table_mode == "n_refs":
        for cl in ALL_CLASSIFIERS:
            data[cl] = {}
            for s_key, s_info in TTA_STRATEGIES.items():
                for rfs in s_info["axis"]:
                    key = f"{s_key}_nrefs{rfs}"
                    data[cl][key] = load_transition_stats(
                        results_dir, dataset, s_key, cl, rfs, seed=seed, against=against
                    )

    elif "hybrid" in table_mode:
        pass 

    elif "compact" ==  table_mode:
        if dataset != "imagenet":
            del strategy_keys[0]
        for cl in TARGET_CLASSIFIERS:
            data[cl] = {}
             
            data[cl] = {}
            for strategy_key in strategy_keys:
                rfs = rfs_values.get(strategy_key)
                if dataset == "imagenet":
                    rfs = 16
                data[cl][strategy_key] = load_transition_stats(
                    results_dir, dataset, strategy_key, cl, rfs, seed=seed, against=against
                )

    elif "calibration" == table_mode:
        if dataset != "imagenet":
            del strategy_keys[0]
        for cl in TARGET_CLASSIFIERS:
            data[cl] = {}
             
            data[cl] = {}
            for strategy_key in strategy_keys:
                rfs = rfs_values.get(strategy_key)
                if dataset == "imagenet":
                    rfs = 16
                data[cl][strategy_key] = load_transition_stats(
                    results_dir, dataset, strategy_key, cl, rfs, seed=seed, against=against
                )
            data[cl]["baseline"] = get_baseline_results(cl, dataset, split)

    else:
        for cl in ALL_CLASSIFIERS:
            data[cl] = {}
            for strategy_key in strategy_keys:
                rfs = rfs_values.get(strategy_key)
                data[cl][strategy_key] = load_transition_stats(
                    results_dir, dataset, strategy_key, cl, rfs, seed=seed, against=against
                )

    # ------------------------------------------------------------------
    # Build LaTeX
    # ------------------------------------------------------------------

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{")

    if table_mode == "comprehensive":

        strategy_key = strategy_keys[0]
        rfs = rfs_values[strategy_key]

        lines.append(r"\begin{tabular}{llcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Backbone} & \textbf{Baseline (\%)} & \textbf{Strategy} & "
            r"\textbf{TTA (\%$\uparrow$)} & "
            r"\textbf{$\Delta$ (95\% CI)} & \textbf{$p$}\\"
        )
        lines.append(r"\midrule")
        # Determine maximum tta_acc for the current classifier across strategies
        valid_accs = [
            data[cl][k]['tta_acc'] 
            for k in strategy_keys 
            if data[cl].get(k) is not None
        ]
        max_acc = max(valid_accs) if valid_accs else None
        for cl_idx, cl in enumerate(ALL_CLASSIFIERS):
            cl_name = latex_classifier_name(cl)

            # Find the maximum tta_acc for the current classifier
            valid_accs = [
                data[cl][sk]["tta_acc"]
                for sk in strategy_keys
                if data[cl][sk] is not None
            ]
            max_acc = max(valid_accs) if valid_accs else None

            for s_idx, strategy_key in enumerate(strategy_keys):
                r = data[cl][strategy_key]

                auc = f"{r["metrics"]["auc"]:.2f}"
                ece = f"{r["metrics"]["ece"]:.2f}"
                if r is None:
                    lines.append(
                        " & ".join([latex_classifier_name(cl)] + ["--"] * 8) + r" \\"
                    )
                    continue

                classifier_cell = (
                    f"\\multirow{{{len(strategy_keys)}}}{{*}}{{{cl_name}}}"
                    if s_idx == 0
                    else ""
                )
                baseline_cell = (
                    f"\\multirow{{{len(strategy_keys)}}}{{*}}{{{r['baseline_acc']:.2f}}}"
                    if s_idx == 0
                    else ""
                )
                delta_prefix = "+" if r["delta_pp"] >= 0 else ""
                delta_ci = f"{delta_prefix}{r['delta_pp']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"

                # Format and apply \textbf if value matches the maximum
                acc_str = f"{r['tta_acc']:.2f}"
                if max_acc is not None and r["tta_acc"] == max_acc:
                    acc_str = f"\\textbf{{{acc_str}}}"

                row = [
                    classifier_cell,
                    baseline_cell,
                    strategy_label(strategy_key),
                    acc_str,
                    delta_ci,
                    format_pvalue(r["p_value"]),
                    ece,
                    auc
                ]
                lines.append(" & ".join(row) + r" \\")

            if cl_idx < len(ALL_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    elif table_mode == "benchmarks":
        lines.append(r"\begin{tabular}{llcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Backbone} & \textbf{Benchmark} & \textbf{Baseline (\%)} & "
            r"\textbf{Strategy} & \textbf{TTA (\%$\uparrow$)} & "
            r"\textbf{$\Delta$ (95\% CI)} & \textbf{$p$}"
            r" \\"
        )
        lines.append(r"\midrule")

        for cl_idx, cl in enumerate(TARGET_CLASSIFIERS):
            cl_name = latex_escape(latex_classifier_name(cl))

            printable_rows = []
            for ds, ds_split in ALL_SPLITS.items():
                for strategy_key in strategy_keys:
                    if strategy_key == "ablation/adain_tta":
                        continue
                    if data[cl].get(ds, {}).get(strategy_key) is not None:
                        printable_rows.append((ds, ds_split, strategy_key))

            total_cl_rows = len(printable_rows)
            cl_rendered = False

            valid_ds_keys = list(ALL_SPLITS.keys())
            for ds_idx, (ds, ds_split) in enumerate(ALL_SPLITS.items()):
                ds_rows = [r_tuple for r_tuple in printable_rows if r_tuple[0] == ds]
                ds_row_count = len(ds_rows)
                if ds_row_count == 0:
                    continue
                ds_rendered = False

                # Calculate max accuracy for the current dataset and classifier
                ds_accs = [
                    data[cl][ds][sk]["tta_acc"]
                    for _, _, sk in ds_rows
                    if "tta_acc" in data[cl][ds][sk]
                ]
                ds_max_acc = max(ds_accs) if ds_accs else None

                for r_ds, r_split, strategy_key in ds_rows:
                    r = data[cl][r_ds][strategy_key]

                    classifier_cell = (
                        f"\\multirow{{{total_cl_rows}}}{{*}}{{{cl_name}}}"
                        if not cl_rendered
                        else ""
                    )
                    cl_rendered = True

                    clean_split = latex_escape(r_split)
                    dataset_cell = (
                        f"\\multirow{{{ds_row_count}}}{{*}}{{{clean_split}}}"
                        if not ds_rendered
                        else ""
                    )

                    baseline_cell = (
                        f"\\multirow{{{ds_row_count}}}{{*}}{{{r['baseline_acc']:.2f}}}"
                        if not ds_rendered
                        else ""
                    )
                    ds_rendered = True

                    delta_prefix = "+" if r["delta_pp"] >= 0 else ""
                    delta_ci = f"{delta_prefix}{r['delta_pp']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"

                    acc_str = f"{r['tta_acc']:.2f}"
                    if ds_max_acc is not None and r["tta_acc"] == ds_max_acc:
                        acc_str = f"\\textbf{{{acc_str}}}"

                    clean_strat = latex_escape(strategy_label(strategy_key))

                    row = [
                        classifier_cell,
                        dataset_cell,
                        baseline_cell,
                        clean_strat,
                        acc_str,
                        delta_ci,
                        format_pvalue(r["p_value"]),
                    ]
                    lines.append(" & ".join(row) + r" \\")

                # Insert partial line between datasets (columns 2 through 7)
                if ds_idx < len(valid_ds_keys) - 1:
                    lines.append(r"\cmidrule{2-7}")

            if cl_idx < len(TARGET_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")

    elif table_mode == "n_refs":
        del TTA_STRATEGIES["hybrid_tta"]
        # Union of all axis values sorted to establish complete columns
        all_refs = sorted(
            {rfs for strat in TTA_STRATEGIES.values() for rfs in strat["axis"]}
        )
        ref_cols = [f"$N={rfs}$" for rfs in all_refs]

        # Added an extra 'c' for the delta column
        lines.append(r"\begin{tabular}{llc" + "c" * (len(ref_cols) + 1) + "}")
        lines.append(r"\toprule")
        lines.append(
            r" & & & \multicolumn{"
            + str(len(ref_cols))
            + r"}{c}{\textbf{TTA ($N_{\text{refs}}$)}} & \\"
        )
        lines.append(f"\\cmidrule(lr){{4-{3 + len(ref_cols)}}}")
        lines.append(
            r"\textbf{Classifier} & \textbf{Strategy} & \textbf{Baseline} & "
            + " & ".join(ref_cols)
            + r" & \textbf{$\Delta (N_{\min} \rightarrow N_{\max})$} \\"
        )
        lines.append(r"\midrule")

        num_strats = len(TTA_STRATEGIES)

        for cl_idx, cl in enumerate(ALL_CLASSIFIERS):
            cl_name = latex_classifier_name(cl)

            # Find global max tta_acc across strategies
            all_valid_accs = []
            for s_key in TTA_STRATEGIES:
                for rfs in TTA_STRATEGIES[s_key]["axis"]:
                    data_entry = data.get(cl, {}).get(f"{s_key}_nrefs{rfs}")
                    if data_entry and data_entry.get("tta_acc") is not None:
                        all_valid_accs.append(data_entry["tta_acc"])
            max_acc = max(all_valid_accs) if all_valid_accs else None

            for s_idx, (s_key, s_info) in enumerate(TTA_STRATEGIES.items()):
                strat_label = s_info.get("label", s_key)
                strat_axis = sorted(s_info["axis"])

                baseline_val = None
                for rfs in strat_axis:
                    entry = data.get(cl, {}).get(f"{s_key}_nrefs{rfs}")
                    if entry and entry.get("baseline_acc") is not None:
                        baseline_val = entry["baseline_acc"]
                        break

                cl_cell = (
                    f"\\multirow{{{num_strats}}}{{*}}{{{cl_name}}}"
                    if s_idx == 0
                    else ""
                )
                bs_cell = (
                    f"\\multirow{{{num_strats}}}{{*}}{{{baseline_val:.2f}}}"
                    if s_idx == 0 and baseline_val is not None
                    else ""
                )

                ref_cells = []
                for rfs in all_refs:
                    if rfs not in strat_axis:
                        ref_cells.append("--")
                        continue

                    r = data.get(cl, {}).get(f"{s_key}_nrefs{rfs}")
                    if r is None or r.get("tta_acc") is None:
                        ref_cells.append("--")
                    else:
                        val_str = f"{r['tta_acc']:.2f}"
                        if max_acc is not None and r["tta_acc"] == max_acc:
                            val_str = f"\\textbf{{{val_str}}}"
                        ref_cells.append(val_str)

                # Compute delta from lowest axis entry to highest axis entry for this strategy
                min_r = data.get(cl, {}).get(f"{s_key}_nrefs{strat_axis[0]}")
                max_r = data.get(cl, {}).get(f"{s_key}_nrefs{strat_axis[-1]}")

                if (
                    min_r
                    and max_r
                    and min_r.get("tta_acc") is not None
                    and max_r.get("tta_acc") is not None
                ):
                    delta = max_r["tta_acc"] - min_r["tta_acc"]
                    delta_str = f"{'+' if delta >= 0 else ''}{delta:.2f}"
                else:
                    delta_str = "--"

                row = [cl_cell, strat_label, bs_cell] + ref_cells + [delta_str]
                lines.append(" & ".join(row) + r" \\")

            if cl_idx < len(ALL_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    elif table_mode == "hybrid":
        views = hybrid_views
        splits_to_compare = hybrid_splits
        sty = hybrid_sty

        # ("geo", ...) uses the plain geometric_tta strategy (no split/sty axis).
        # ("hybrid_s{sp}", ...) uses strategy_key="hybrid" with a fixed sty and
        # varying split, at each ensemble size (view = rfs = nr).
        configs = [("geo", None, "Geo-TTA only")]
        for sp in splits_to_compare:
            configs.append((f"hybrid_s{sp}", sp, f"Hybrid (sty={sty}, split={sp})"))

        data = {}
        for cl in ALL_CLASSIFIERS:
            data[cl] = {}
            for cfg_key, sp, _ in configs:
                if cfg_key == "geo":
                    data[cl][cfg_key] = {
                        view: load_transition_stats(
                            results_dir, dataset, "geometric_tta", cl, view, seed=seed, against=against
                        )
                        for view in views
                    }
                else:
                    data[cl][cfg_key] = {
                        view: load_transition_stats(
                            results_dir, dataset, "hybrid_tta", cl, view,
                            seed=seed, against=against,split=sp, sty=sty,
                        )
                        for view in views
                    }

        def stars(p):
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return ""
            if p < 0.001:
                return r"$^{***}$"
            if p < 0.01:
                return r"$^{**}$"
            if p < 0.05:
                return r"$^{*}$"
            return ""

        lines.append(r"\small")
        lines.append(r"\begin{tabular}{ll" + "c" * len(views) + "}")
        lines.append(r"\toprule")
        lines.append(
            r" & & \multicolumn{" + str(len(views)) + r"}{c}{\textbf{Total Ensemble Size}} \\"
        )
        lines.append(rf"\cmidrule(lr){{3-{2 + len(views)}}}")
        lines.append(
            r"\textbf{Backbone} & \textbf{Configuration} & "
            + " & ".join(f"\\textbf{{{v}}}" for v in views)
            + r" \\"
        )
        lines.append(r"\midrule")

        num_configs = len(configs)

        for cl_idx, cl in enumerate(ALL_CLASSIFIERS):
            cl_name = latex_classifier_name(cl)

            best_by_view = {}
            for view in views:
                accs = [
                    data[cl][cfg_key][view]["tta_acc"]
                    for cfg_key, _, _ in configs
                    if data[cl][cfg_key][view] is not None
                ]
                best_by_view[view] = max(accs) if accs else None

            for cfg_idx, (cfg_key, _, cfg_label) in enumerate(configs):
                backbone_cell = (
                    f"\\multirow{{{num_configs}}}{{*}}{{{cl_name}}}"
                    if cfg_idx == 0
                    else ""
                )
                row = [backbone_cell, cfg_label]

                for view in views:
                    r = data[cl][cfg_key][view]
                    if r is None:
                        row.append("--")
                        continue

                    val_str = f"{r['tta_acc']:.2f}{stars(r['p_value'])}"
                    if best_by_view[view] is not None and r["tta_acc"] == best_by_view[view]:
                        val_str = f"\\textbf{{{val_str}}}"
                    row.append(val_str)

                lines.append(" & ".join(row) + r" \\")

            if cl_idx < len(ALL_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{"
            + str(2 + len(views))
            + r"}{l}{\footnotesize $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ "
            r"(McNemar test vs.\ baseline)} \\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    elif table_mode == "hybrid_overview":
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Backbone} & \textbf{Baseline} & \textbf{Geo-TTA ($N=32$)} & "
            r"\textbf{StyleID only (best $N$)} & \textbf{Hybrid S1 ($N=32$)} & "
            r"\textbf{Hybrid S2 ($N=32$)} & \textbf{$\Delta$ S2 vs Geo} \\"
        )
        lines.append(r"\midrule")

        for cl_idx, cl in enumerate(ALL_CLASSIFIERS):
            cl_name = latex_classifier_name(cl)
            
            # Load statistics for all 5 configurations
            geo = load_transition_stats(
                results_dir, dataset, "geometric_tta", cl, 64, seed=seed, against=against
            )
            # Fetch optimal N for StyleID
            style_rfs = rfs_values.get("ablation/retristyle", 16)
            style_id = load_transition_stats(
                results_dir, dataset, "ablation/retristyle", cl, style_rfs, seed=seed, against=against
            )
            h_s1 = load_transition_stats(
                results_dir, dataset, "hybrid_tta", cl, 64, seed=seed, against=against, split=1, sty=hybrid_sty
            )
            h_s2 = load_transition_stats(
                results_dir, dataset, "hybrid_tta", cl, 64, seed=seed, against=against, split=4, sty=hybrid_sty
            )

            baseline_val = geo["baseline_acc"] if geo else 0.0
            geo_acc = geo["tta_acc"] if geo else 0.0
            style_acc = style_id["tta_acc"] if style_id else 0.0
            hs1_acc = h_s1["tta_acc"] if h_s1 else 0.0
            hs2_acc = h_s2["tta_acc"] if h_s2 else 0.0

            # Determine best performing strategy per classifier
            acc_list = [geo_acc, style_acc, hs1_acc, hs2_acc]
            max_acc = max(acc_list)

            def fmt(acc):
                s = f"{acc:.2f}"
                return f"\\textbf{{{s}}}" if acc == max_acc else s

            delta_s2_geo = hs2_acc - geo_acc
            delta_str = f"{'+' if delta_s2_geo >= 0 else ''}{delta_s2_geo:.2f}"

            row = [
                cl_name,
                f"{baseline_val:.2f}",
                fmt(geo_acc),
                fmt(style_acc),
                fmt(hs1_acc),
                fmt(hs2_acc),
                delta_str,
            ]
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    elif table_mode == "hybrid_calibration":
        target_classifiers = [
            "densenet121",
            "swin_base_patch4_window7_224",
            "dinov2_vitb14",
        ]

        lines.append(r"\begin{tabular}{llcccccccc}")
        lines.append(r"\toprule")
        lines.append(
            r" & & \multicolumn{2}{c}{\textbf{Geo-TTA ($N=64$)}} & "
            r"\multicolumn{2}{c}{\textbf{StyleID ($N=16$)}} & "
            r"\multicolumn{2}{c}{\textbf{Hybrid S1 ($N=64$)}} & "
            r"\multicolumn{2}{c}{\textbf{Hybrid S2 ($N=64$)}} \\"
        )
        lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}")
        lines.append(
            r"\textbf{Dataset} & \textbf{Backbone} & "
            r"\textbf{ECE $\downarrow$} & \textbf{AUC $\uparrow$} & "
            r"\textbf{ECE $\downarrow$} & \textbf{AUC $\uparrow$} & "
            r"\textbf{ECE $\downarrow$} & \textbf{AUC $\uparrow$} & "
            r"\textbf{ECE $\downarrow$} & \textbf{AUC $\uparrow$} \\"
        )
        lines.append(r"\midrule")

        datasets_to_process = list(ALL_SPLITS.keys())
        num_cl = len(target_classifiers)

        for ds_idx, ds_name in enumerate(datasets_to_process):
            ds_label = latex_escape(ALL_DATASETS.get(ds_name, ds_name))

            for cl_idx, cl in enumerate(target_classifiers):
                cl_name = latex_classifier_name(cl)

                geo = load_transition_stats(
                    results_dir, ds_name, "geometric_tta", cl, 64, seed=seed, against=against
                )
                style_rfs = rfs_values.get("ablation/retristyle", 16)
                if ds_name == "imagenet":
                    style_rfs = 16

                style_id = load_transition_stats(
                    results_dir, ds_name, "ablation/retristyle", cl, style_rfs, seed=seed, against=against
                )
                h_s1 = load_transition_stats(
                    results_dir, ds_name, "hybrid_tta", cl, 64, seed=seed, against=against, split=1, sty=hybrid_sty
                )
                h_s2 = load_transition_stats(
                    results_dir, ds_name, "hybrid_tta", cl, 64, seed=seed, against=against, split=4, sty=hybrid_sty
                )

                methods = [geo, style_id, h_s1, h_s2]

                def extract_val(res, key):
                    if res and res.get("metrics") and key in res["metrics"]:
                        val = res["metrics"][key]
                        return float(val) if not np.isnan(val) else None
                    return None

                ece_vals = [extract_val(m, "ece") for m in methods]
                auc_vals = [extract_val(m, "auc") for m in methods]

                valid_eces = [v for v in ece_vals if v is not None]
                valid_aucs = [v for v in auc_vals if v is not None]

                best_ece = min(valid_eces) if valid_eces else None
                best_auc = max(valid_aucs) if valid_aucs else None

                def format_cell(val, best_val):
                    if val is None:
                        return "--"
                    val_str = f"{val:.2f}"
                    if best_val is not None and np.isclose(val, best_val, atol=1e-5):
                        return f"\\textbf{{{val_str}}}"
                    return val_str

                dataset_cell = (
                    f"\\multirow{{{num_cl}}}{{*}}{{{ds_label}}}"
                    if cl_idx == 0
                    else ""
                )

                row = [dataset_cell, cl_name]
                for ece, auc in zip(ece_vals, auc_vals):
                    row.append(format_cell(ece, best_ece))
                    row.append(format_cell(auc, best_auc))

                lines.append(" & ".join(row) + r" \\")

            if ds_idx < len(datasets_to_process) - 1:
                lines.append(r"\cmidrule{2-10}")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    elif table_mode == "hybrid_overview_all":
        target_classifiers = [
            "densenet121",
            "swin_base_patch4_window7_224",
            "dinov2_vitb14",
        ]

        lines.append(r"\begin{tabular}{llcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Dataset} & \textbf{Backbone} & \textbf{Baseline} & "
            r"\textbf{Geo-TTA ($N=32$)} & \textbf{StyleID only (best $N$)} & "
            r"\textbf{Hybrid S1 ($N=32$)} & \textbf{Hybrid S2 ($N=32$)} & "
            r"\textbf{$\Delta$ S2 vs Geo} \\"
        )
        lines.append(r"\midrule")

        datasets_to_process = list(ALL_SPLITS.keys())
        num_cl = len(target_classifiers)

        for ds_idx, ds_name in enumerate(datasets_to_process):
            ds_label = latex_escape(ALL_DATASETS.get(ds_name, ds_name))

            for cl_idx, cl in enumerate(target_classifiers):
                cl_name = latex_classifier_name(cl)

                # Load statistics across all 4 evaluation pathways
                geo = load_transition_stats(
                    results_dir, ds_name, "geometric_tta", cl, 32, seed=seed, against=against
                )
                style_rfs = rfs_values.get("ablation/retristyle", 16)
                if ds_name == "imagenet":
                    style_rfs = 16

                style_id = load_transition_stats(
                    results_dir, ds_name, "ablation/retristyle", cl, style_rfs, seed=seed, against=against
                )
                h_s1 = load_transition_stats(
                    results_dir, ds_name, "hybrid_tta", cl, 32, seed=seed, against=against, split=1, sty=hybrid_sty
                )
                h_s2 = load_transition_stats(
                    results_dir, ds_name, "hybrid_tta", cl, 32, seed=seed, against=against, split=4, sty=hybrid_sty
                )

                baseline_val = geo["baseline_acc"] if geo else 0.0
                geo_acc = geo["tta_acc"] if geo else 0.0
                style_acc = style_id["tta_acc"] if style_id else 0.0
                hs1_acc = h_s1["tta_acc"] if h_s1 else 0.0
                hs2_acc = h_s2["tta_acc"] if h_s2 else 0.0

                acc_list = [geo_acc, style_acc, hs1_acc, hs2_acc]
                max_acc = max(acc_list) if acc_list else None

                def fmt(acc):
                    s = f"{acc:.2f}"
                    return f"\\textbf{{{s}}}" if (max_acc is not None and acc == max_acc) else s

                delta_s2_geo = hs2_acc - geo_acc
                delta_str = f"{'+' if delta_s2_geo >= 0 else ''}{delta_s2_geo:.2f}"

                ds_cell = (
                    f"\\multirow{{{num_cl}}}{{*}}{{{ds_label}}}"
                    if cl_idx == 0
                    else ""
                )

                row = [
                    ds_cell,
                    cl_name,
                    f"{baseline_val:.2f}",
                    fmt(geo_acc),
                    fmt(style_acc),
                    fmt(hs1_acc),
                    fmt(hs2_acc),
                    delta_str,
                ]
                lines.append(" & ".join(row) + r" \\")

            if ds_idx < len(datasets_to_process) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    elif table_mode == "compact":  # comprehensive

        strategy_key = strategy_keys[0]
        rfs = rfs_values[strategy_key]

        lines.append(r"\begin{tabular}{llcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Backbone} & \textbf{Baseline (\%)} & \textbf{Strategy} & "
            r"\textbf{TTA (\%$\uparrow$)} & "
            r"\textbf{$\Delta$ (95\% CI)} & \textbf{$p$}\\"
        )
        lines.append(r"\midrule")
        # Determine maximum tta_acc for the current classifier across strategies
        valid_accs = [
            data[cl][k]['tta_acc'] 
            for k in strategy_keys 
            if data[cl].get(k) is not None
        ]
        max_acc = max(valid_accs) if valid_accs else None
        for cl_idx, cl in enumerate(TARGET_CLASSIFIERS):
            cl_name = latex_classifier_name(cl)

            # Find the maximum tta_acc for the current classifier
            valid_accs = [
                data[cl][sk]["tta_acc"]
                for sk in strategy_keys
                if data[cl].get(sk) is not None
            ]
            max_acc = max(valid_accs) if valid_accs else None

            for s_idx, strategy_key in enumerate(strategy_keys):
                r = data[cl].get(strategy_key, None)
                if r is None:
                    continue
                auc = f"{r["metrics"]["auc"]:.2f}"
                ece = f"{r["metrics"]["ece"]:.2f}"
                if r is None:
                    lines.append(
                        " & ".join([latex_classifier_name(cl)] + ["--"] * 8) + r" \\"
                    )
                    continue

                classifier_cell = (
                    f"\\multirow{{{len(strategy_keys)}}}{{*}}{{{cl_name}}}"
                    if s_idx == 0
                    else ""
                )
                baseline_cell = (
                    f"\\multirow{{{len(strategy_keys)}}}{{*}}{{{r['baseline_acc']:.2f}}}"
                    if s_idx == 0
                    else ""
                )
                delta_prefix = "+" if r["delta_pp"] >= 0 else ""
                delta_ci = f"{delta_prefix}{r['delta_pp']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"

                # Format and apply \textbf if value matches the maximum
                acc_str = f"{r['tta_acc']:.2f}"
                if max_acc is not None and r["tta_acc"] == max_acc:
                    acc_str = f"\\textbf{{{acc_str}}}"

                row = [
                    classifier_cell,
                    baseline_cell,
                    strategy_label(strategy_key),
                    acc_str,
                    delta_ci,
                    format_pvalue(r["p_value"]),
                ]
                lines.append(" & ".join(row) + r" \\")

            if cl_idx < len(ALL_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    elif table_mode == "calibration": 

        lines.append(r"\begin{tabular}{llcccc}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Backbone} & \textbf{Method} & \textbf{Baseline ECE} & "
            r"\textbf{Baseline AUC} & \textbf{ECE $\downarrow$} & \textbf{AUC $\uparrow$} \\"
        )
        lines.append(r"\midrule")

        for cl_idx, cl in enumerate(TARGET_CLASSIFIERS):
            cl_name = latex_classifier_name(cl)

            # Compute optimal metrics per classifier for bolding
            valid_ece = [
                data[cl][sk]["metrics"]["ece"]
                for sk in strategy_keys
                if data[cl].get(sk) is not None
            ]
            valid_auc = [
                data[cl][sk]["metrics"]["auc"]
                for sk in strategy_keys
                if data[cl].get(sk) is not None
            ]
            
            min_ece = min(valid_ece) if valid_ece else None
            max_auc = max(valid_auc) if valid_auc else None

            num_strategies = len(strategy_keys)

            for s_idx, strategy_key in enumerate(strategy_keys):
                r = data[cl].get(strategy_key, None)
                if r is None:
                    continue

                ece_val = r["metrics"]["ece"]
                auc_val = r["metrics"]["auc"]

                ece_str = f"{ece_val:.2f}"
                auc_str = f"{auc_val:.2f}"

                if min_ece is not None and ece_val == min_ece:
                    ece_str = f"\\textbf{{{ece_str}}}"
                if max_auc is not None and auc_val == max_auc:
                    auc_str = f"\\textbf{{{auc_str}}}"

                classifier_cell = (
                    f"\\multirow{{{num_strategies}}}{{*}}{{{cl_name}}}"
                    if s_idx == 0
                    else ""
                )
                baseline_ece_cell = (
                    f"\\multirow{{{num_strategies}}}{{*}}{{{data[cl]['baseline']['ece']:.2f}}}"
                    if s_idx == 0
                    else ""
                )
                baseline_auc_cell = (
                    f"\\multirow{{{num_strategies}}}{{*}}{{{data[cl]['baseline']['auc']:.2f}}}"
                    if s_idx == 0
                    else ""
                )

                row = [
                    classifier_cell,
                    strategy_label(strategy_key),
                    baseline_ece_cell,
                    baseline_auc_cell,
                    ece_str,
                    auc_str,
                ]
                lines.append(" & ".join(row) + r" \\")

            if cl_idx < len(TARGET_CLASSIFIERS) - 1:
                lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    print("\n" + "=" * 80)
    print("Generated LaTeX transition table")
    print("=" * 80)
    print(latex)
    print("=" * 80)

    if output_dir is not None:
        output_path = output_dir / f"{dataset}_{table_mode}.tex"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_path.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX table saved to: {output_path}")

    return latex

if __name__ == "__main__":
    results_dir = Path("./results")
    output_dir = Path("./output")
    against = "baseline"
    table_mode = "calibration"
    for ds in ["imagenet", "eurosat", "midog", "camelyon17wilds", "epistr"]:
    # 5.3.3 ImageNet-R — comprehensive, all style-transfer strategies
        generate_transition_table(
            results_dir=results_dir,
            dataset=ds,
            split=TRUE_SPLITS[ALL_SPLITS[ds]],
            strategy_keys=["ablation/adain_tta", "ablation/retristyle", "geometric_tta"],
            rfs_values={
                "ablation/adain_tta": 64,
                "ablation/retristyle": 4,
                "geometric_tta": 64,
            },
            table_mode=table_mode,
            output_dir = output_dir / "latex_tables" / against / table_mode,
            against=against,
            caption=f"Evaluation of baseline {ALL_SPLITS[ds]} and TTA methods across CNN, Vision Transformer, and Foundation Model backbones. Bold indicates the best performance per metric."
        )

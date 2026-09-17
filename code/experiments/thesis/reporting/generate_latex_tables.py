"""
LaTeX Table Generation for Thesis Results
==========================================

Parses the JSON result files from all experiment phases and generates
formatted LaTeX tables ready for inclusion in the thesis document.

Supported table types:
    1. **Style Transfer Comparison** — all methods ranked by quality metrics
    2. **Geometric TTA Baselines** — per-classifier accuracy across seeds
    3. **Ablation: Retrieval Strategy** — accuracy by retrieval method
    4. **Ablation: Aggregation Strategy** — accuracy by eval strategy
    5. **Ablation: n_refs Sweep** — accuracy vs number of references
    6. **Hybrid TTA** — geo/style mixing ratios performance

Usage::

    python -m experiments.reporting.generate_latex_tables \\
        --results_dir ./results \\
        --output_dir ./tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =========================================================================
# Helpers
# =========================================================================
def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_json_files(directory: Path, pattern: str = "*.json") -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def _mean_std_str(values: List[float], fmt: str = ".2f") -> str:
    """Format mean ± std for LaTeX."""
    if not values:
        return "—"
    m = np.mean(values)
    if len(values) > 1:
        s = np.std(values)
        return f"${m:{fmt}} \\pm {s:{fmt}}$"
    return f"${m:{fmt}}$"


def _bold_best(values: Dict[str, str], best_key: str) -> Dict[str, str]:
    """Make the best value bold in LaTeX."""
    result = dict(values)
    if best_key in result:
        result[best_key] = f"\\textbf{{{result[best_key]}}}"
    return result


# =========================================================================
# Table 1: Style Transfer Method Comparison
# =========================================================================
def generate_style_transfer_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table comparing all style transfer methods."""
    results_file = results_dir / "style_transfer_eval" / "all_methods_comparison.json"
    if not results_file.exists():
        print(f"  [skip] {results_file} not found")
        return

    data = _load_json(results_file)

    rows = []
    for method, metrics in sorted(data.items()):
        if "error" in metrics:
            continue
        rows.append({
            "method": method.replace("_", "\\_"),
            "category": metrics.get("category", "?").replace("_", "\\_"),
            "wasserstein": metrics.get("wasserstein_mean", float("inf")),
            "ssim": metrics.get("ssim_mean", 0),
            "lpips": metrics.get("lpips_mean", float("inf")),
            "edge": metrics.get("edge_similarity_mean", 0),
        })

    # Sort by composite score: high SSIM + high edge - low LPIPS
    rows.sort(key=lambda r: r["ssim"] + r["edge"] - r["lpips"], reverse=True)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Style transfer method comparison (20 content $\times$ 40 style images).}",
        r"  \label{tab:style_transfer_comparison}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    Method & Category & Wass.$\downarrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & Edge$\uparrow$ \\",
        r"    \midrule",
    ]
    for r in rows:
        lines.append(
            f"    {r['method']} & {r['category']} & "
            f"{r['wasserstein']:.4f} & {r['ssim']:.4f} & "
            f"{r['lpips']:.4f} & {r['edge']:.4f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "style_transfer_comparison.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 2: Geometric TTA Baselines
# =========================================================================
def generate_geometric_tta_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table for geometric TTA baselines across classifiers."""
    geo_dir = results_dir / "geometric_tta"
    if not geo_dir.exists():
        geo_dir = results_dir / "geometric_tta"
    files = _find_json_files(geo_dir, "*_results.json")
    if not files:
        print(f"  [skip] No geometric TTA results in {geo_dir}")
        return

    # Group by classifier × eval_strategy
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        clf = data.get("classifier", "?")
        strat = data.get("eval_strategy", "?")
        acc = data.get("metrics", {}).get("accuracy")
        if acc is not None:
            grouped.setdefault((clf, strat), []).append(acc * 100)

    if not grouped:
        print("  [skip] No geometric TTA results parsed")
        return

    classifiers = sorted(set(k[0] for k in grouped))
    strategies = sorted(set(k[1] for k in grouped))

    header = " & ".join(["Classifier"] + [s.capitalize() for s in strategies])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Geometric TTA baselines (Accuracy \%).}",
        r"  \label{tab:geometric_tta_baselines}",
        f"  \\begin{{tabular}}{{l{'c' * len(strategies)}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    for clf in classifiers:
        cells = [clf.replace("_", "\\_")]
        for strat in strategies:
            vals = grouped.get((clf, strat), [])
            cells.append(_mean_std_str(vals))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "geometric_tta_baselines.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 3: Retrieval Strategy Ablation
# =========================================================================
def generate_retrieval_ablation_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table for retrieval strategy ablation."""
    abl_dir = results_dir / "thesis" / "ablation"
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json_files(abl_dir, "*_results.json")
    if not files:
        print(f"  [skip] No ablation results in {abl_dir}")
        return

    # Group by retrieval_strategy × classifier
    grouped: Dict[Tuple[str, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        retr = data.get("retrieval_strategy")
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        if retr and acc is not None:
            grouped.setdefault((retr, clf), []).append(acc * 100)

    if not grouped:
        print("  [skip] No retrieval ablation results")
        return

    strategies = sorted(set(k[0] for k in grouped))
    classifiers = sorted(set(k[1] for k in grouped))

    header = " & ".join(["Retrieval"] + [c.replace("_", "\\_") for c in classifiers])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Retrieval strategy ablation (Accuracy \%).}",
        r"  \label{tab:retrieval_ablation}",
        f"  \\begin{{tabular}}{{l{'c' * len(classifiers)}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    for strat in strategies:
        cells = [strat.replace("_", "\\_")]
        for clf in classifiers:
            vals = grouped.get((strat, clf), [])
            cells.append(_mean_std_str(vals))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "retrieval_ablation.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 4: Aggregation Strategy Ablation
# =========================================================================
def generate_eval_ablation_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table for aggregation (eval strategy) ablation."""
    abl_dir = results_dir / "thesis" / "ablation"
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json_files(abl_dir, "*_results.json")
    if not files:
        print(f"  [skip] No ablation results in {abl_dir}")
        return

    grouped: Dict[Tuple[str, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        strat = data.get("eval_strategy")
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        if strat and acc is not None:
            grouped.setdefault((strat, clf), []).append(acc * 100)

    if not grouped:
        return

    strategies = sorted(set(k[0] for k in grouped))
    classifiers = sorted(set(k[1] for k in grouped))

    header = " & ".join(["Aggregation"] + [c.replace("_", "\\_") for c in classifiers])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Aggregation strategy ablation (Accuracy \%).}",
        r"  \label{tab:eval_ablation}",
        f"  \\begin{{tabular}}{{l{'c' * len(classifiers)}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    for strat in strategies:
        cells = [strat.replace("_", "\\_")]
        for clf in classifiers:
            vals = grouped.get((strat, clf), [])
            cells.append(_mean_std_str(vals))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "eval_ablation.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 5: n_refs Sweep
# =========================================================================
def generate_nrefs_sweep_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table for n_refs sweep."""
    abl_dir = results_dir / "thesis" / "ablation"
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json_files(abl_dir, "*_results.json")
    if not files:
        print(f"  [skip] No ablation results in {abl_dir}")
        return

    grouped: Dict[Tuple[int, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        n_refs = data.get("n_refs")
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        if n_refs is not None and acc is not None:
            grouped.setdefault((n_refs, clf), []).append(acc * 100)

    if not grouped:
        return

    nrefs_vals = sorted(set(k[0] for k in grouped))
    classifiers = sorted(set(k[1] for k in grouped))

    header = " & ".join(["$n_{refs}$"] + [c.replace("_", "\\_") for c in classifiers])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Number of style-transfer references ablation (Accuracy \%).}",
        r"  \label{tab:nrefs_sweep}",
        f"  \\begin{{tabular}}{{r{'c' * len(classifiers)}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    for nr in nrefs_vals:
        cells = [str(nr)]
        for clf in classifiers:
            vals = grouped.get((nr, clf), [])
            cells.append(_mean_std_str(vals))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "nrefs_sweep.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 6: Hybrid TTA
# =========================================================================
def generate_hybrid_tta_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table for hybrid TTA mixing ratios."""
    hybrid_dir = results_dir / "thesis" / "hybrid_tta"
    if not hybrid_dir.exists():
        hybrid_dir = results_dir / "hybrid_tta"
    files = _find_json_files(hybrid_dir, "*_results.json")
    if not files:
        print(f"  [skip] No hybrid TTA results in {hybrid_dir}")
        return

    grouped: Dict[Tuple[float, str], List[float]] = {}
    ece_grouped: Dict[Tuple[float, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        geo = data.get("geo_frac")
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        ece = data.get("metrics", {}).get("ece")
        if geo is not None and acc is not None:
            grouped.setdefault((geo, clf), []).append(acc * 100)
        if geo is not None and ece is not None:
            ece_grouped.setdefault((geo, clf), []).append(ece * 100)

    if not grouped:
        return

    ratios = sorted(set(k[0] for k in grouped))
    classifiers = sorted(set(k[1] for k in grouped))

    header = " & ".join(["Geo/Style"] + [c.replace("_", "\\_") for c in classifiers])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Hybrid TTA: geometric/style-transfer mixing ratios (Accuracy \%).}",
        r"  \label{tab:hybrid_tta}",
        f"  \\begin{{tabular}}{{l{'c' * len(classifiers)}}}",
        r"    \toprule",
        f"    {header} \\\\",
        r"    \midrule",
    ]
    for geo in ratios:
        style = 1.0 - geo
        label = f"{geo:.0%}/{style:.0%}"
        cells = [label]
        for clf in classifiers:
            vals = grouped.get((geo, clf), [])
            cells.append(_mean_std_str(vals))
        lines.append("    " + " & ".join(cells) + " \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "hybrid_tta.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 7: Domain Shift Summary
# =========================================================================
def generate_domain_shift_table(results_dir: Path, output_dir: Path):
    """Generate LaTeX table summarising domain shift metrics across variants."""
    shift_dir = results_dir / "domain_shift_analysis"
    files = _find_json_files(shift_dir, "*_results.json")
    if not files:
        print(f"  [skip] No domain shift results in {shift_dir}")
        return

    rows = []
    for f in files:
        data = _load_json(f)
        tag = data.get("tag", f.stem)
        pm = data.get("pairwise_metrics", {})
        rows.append({
            "tag": tag.replace("_", "\\_"),
            "gram": pm.get("gram_distance_mean"),
            "wasserstein": pm.get("wasserstein_mean"),
            "edge_ssim": pm.get("edge_ssim_mean"),
            "lpips": pm.get("lpips_mean"),
            "mmd": data.get("mmd"),
            "assessment": data.get("shift_assessment", "—"),
        })

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Domain shift characterisation: ImageNet-1k $\to$ variants.}",
        r"  \label{tab:domain_shift_summary}",
        r"  \begin{tabular}{lccccl}",
        r"    \toprule",
        r"    Variant & Gram$\uparrow$ & Wass. & Edge$\uparrow$ & MMD$\downarrow$ & Assessment \\",
        r"    \midrule",
    ]
    for r in rows:
        def _f(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "—"
        lines.append(
            f"    {r['tag']} & {_f(r['gram'])} & {_f(r['wasserstein'])} & "
            f"{_f(r['edge_ssim'])} & {_f(r['mmd'], '.6f')} & {r['assessment']} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    out_path = output_dir / "domain_shift_summary.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Table 8: Per-sample comparison (Geometric vs Style TTA)
# =========================================================================
def generate_persample_comparison_table(results_dir: Path, output_dir: Path):
    """Generate summary of which samples each TTA type gets right/wrong."""
    geo_dir = results_dir / "thesis" / "geometric_tta"
    style_dir = results_dir / "thesis" / "ablation"

    geo_files = _find_json_files(geo_dir, "*_predictions.json")
    style_files = _find_json_files(style_dir, "*_predictions.json")

    if not geo_files or not style_files:
        print("  [skip] Need both geometric and style TTA predictions for comparison")
        return

    # Load first available pair
    geo_data = _load_json(geo_files[0])
    style_data = _load_json(style_files[0])

    geo_preds = geo_data.get("predictions", [])
    style_preds = style_data.get("predictions", [])

    if not geo_preds or not style_preds:
        return

    n = min(len(geo_preds), len(style_preds))
    both_correct = 0
    geo_only = 0
    style_only = 0
    both_wrong = 0

    for i in range(n):
        g_true = geo_preds[i]["y_true"]
        g_pred = np.argmax(geo_preds[i]["y_pred"])
        s_pred = np.argmax(style_preds[i]["y_pred"])
        g_true_val = g_true if isinstance(g_true, int) else g_true[0] if isinstance(g_true, list) else int(g_true)

        g_correct = g_pred == g_true_val
        s_correct = s_pred == g_true_val

        if g_correct and s_correct:
            both_correct += 1
        elif g_correct and not s_correct:
            geo_only += 1
        elif not g_correct and s_correct:
            style_only += 1
        else:
            both_wrong += 1

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Per-sample comparison: geometric vs.\ style-transfer TTA.}",
        r"  \label{tab:persample_comparison}",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        r"    & Style Correct & Style Wrong \\",
        r"    \midrule",
        f"    Geometric Correct & {both_correct} & {geo_only} \\\\",
        f"    Geometric Wrong & {style_only} & {both_wrong} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\\\[6pt] Total: {n} samples. "
        f"Style uniquely correct: {style_only} ({style_only/n*100:.1f}\\%)",
        r"\end{table}",
    ]

    out_path = output_dir / "persample_comparison.tex"
    out_path.write_text("\n".join(lines))
    print(f"  [saved] {out_path}")


# =========================================================================
# Main
# =========================================================================
def generate_all_tables(results_dir: Path, output_dir: Path):
    """Generate all thesis LaTeX tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LaTeX Table Generation")
    print("=" * 60)

    generate_style_transfer_table(results_dir, output_dir)
    generate_geometric_tta_table(results_dir, output_dir)
    generate_retrieval_ablation_table(results_dir, output_dir)
    generate_eval_ablation_table(results_dir, output_dir)
    generate_nrefs_sweep_table(results_dir, output_dir)
    generate_hybrid_tta_table(results_dir, output_dir)
    generate_domain_shift_table(results_dir, output_dir)
    generate_persample_comparison_table(results_dir, output_dir)

    print(f"\nAll tables written to {output_dir}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate LaTeX tables for thesis results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", type=str, default="./results",
                   help="Root results directory")
    p.add_argument("--output_dir", type=str, default="./tables",
                   help="Output directory for .tex files")
    return p


def main():
    args = build_parser().parse_args()
    generate_all_tables(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

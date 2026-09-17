import json
from pathlib import Path
from code.config.constants import ALL_CLASSIFIERS, ALL_DATASETS, DEFAULT_SEED, ALL_METHODS, CLEAN_CLASSIFIERS


def generate_latex_table(
    summary_dir: Path,
    output_tex: Path,
    target_dataset: str,
    target_method: str,
    target_rfs: int = 1,
    target_sty: int = 1,
    target_use_n: int = 1,
    percent: bool = False, 
    seed: int = DEFAULT_SEED,
):
    """Generates a LaTeX table reading individual sample transition JSON files per classifier."""
    
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Sample transitions for \\textbf{{{ALL_DATASETS[target_dataset]}}} using \\textbf{{{ALL_METHODS[target_method]}}} (rfs={target_rfs}).}}",
        f"\\label{{tab:{target_dataset}_{target_method}_rfs{target_rfs}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
    ]
    if percent:
        header = r"\textbf{Classifier} & \textbf{Saved \%} $\uparrow$ & \textbf{Made Neg. \%} $\downarrow$ \\"
    else:
        header = r"\textbf{Classifier} & \textbf{Saved} $\uparrow$ & \textbf{Stayed Pos.} & \textbf{Made Neg.} $\downarrow$ & \textbf{Stayed Neg.} \\"

    latex_lines.extend((header, r"\midrule" ))


    # Build key based on TTA strategy configuration
    if target_method == "hybrid_tta":
        key = f"nr_{target_rfs}_sty_{target_sty}_split_{target_use_n}_seed_{seed}"
    else:
        key = f"rfs_{target_rfs}_seed_{seed}"

    filename = f"{key}_sample_transitions_summary.json"

    for cl in ALL_CLASSIFIERS:
        json_file = summary_dir / target_method / target_dataset / cl / filename
        
        if not json_file.exists():
            print(f"not found {json_file}")
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        counts = data["counts"]
        saved = counts["saved"]
        stayed_pos = counts["stayed_positive"]
        made_neg = counts["made_negative"]
        stayed_neg = counts["stayed_negative"]

        cl_name = CLEAN_CLASSIFIERS[cl][0] if cl in CLEAN_CLASSIFIERS else cl
        if percent:
            sum = made_neg + stayed_neg + saved + stayed_pos
            per_saved = (saved / sum) * 100
            per_made_neg = (made_neg / sum) * 100
            row = f"{cl_name} & \\textbf{{{per_saved:.2f}}}  & {per_made_neg:.2f}\\\\"           
        else:
            row = f"{cl_name} & \\textbf{{{saved}}} & {stayed_pos} & {made_neg} & {stayed_neg} \\\\"
        latex_lines.append(row)

    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, "w") as f:
        f.write("\n".join(latex_lines))


if __name__ == "__main__":
    for i in range(2):
        generate_latex_table(
            summary_dir=Path("./results/sample_analysis"),
            output_tex=Path(f"{i}_filtered_table.tex"),
            target_dataset="epistr",
            target_method="ablation/retristyle",
            target_rfs=4,
            percent=i
        )

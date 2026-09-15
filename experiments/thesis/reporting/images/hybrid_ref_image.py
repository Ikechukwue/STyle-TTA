import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

# ============================================================
# DATA
# ============================================================

classifiers = ["CNN", "ViT", "FM"]

baseline = {
    "CNN": 38.60,
    "ViT": 58.79,
    "FM": 67.47,
}

N_values = [8, 16, 32, 64]

data = {
    "Geo": {
        "CNN": [41.24, 41.50, 41.63, 41.90],
        "ViT": [60.60, 60.94, 60.96, 60.93],
        "FM": [68.68, 68.87, 68.94, 69.03],
    },
    "StyleID": {
        "CNN": [48.67, 49.24, None, None],
        "ViT": [62.87, 63.25, None, None],
        "FM": [66.59, 66.18, None, None],
    },
    "Hybrid S1": {
        "CNN": [47.63, 45.66, 43.69, 42.75],
        "ViT": [64.51, 63.85, 62.76, 61.89],
        "FM": [69.59, 69.70, 69.41, 69.18],
    },
    "Hybrid S2": {
        "CNN": [48.42, 48.89, 49.07, 49.21],
        "ViT": [62.85, 63.02, 63.06, 63.06],
        "FM": [67.66, 67.75, 67.70, 67.63],
    },
}

# ============================================================
# COLORS & OFFSETS
# ============================================================

method_colors = {
    "Geo": "#1f77b4",
    "StyleID": "#2ca02c",
    "Hybrid S1": "#ff7f0e",
    "Hybrid S2": "#d62728",
}

method_offsets = {
    "Geo": -0.15,
    "StyleID": -0.05,
    "Hybrid S1": 0.05,
    "Hybrid S2": 0.15,
}

# ============================================================
# SHADE FUNCTION
# ============================================================


def shade_color(color, N):
    t = (np.log2(N) - np.log2(8)) / (np.log2(64) - np.log2(8))
    rgba = np.array(to_rgba(color))
    strength = 0.35 + 0.65 * t
    rgba[:3] = (1 - strength) * np.ones(3) + strength * rgba[:3]
    return rgba


# ============================================================
# LEGEND HANDLES
# ============================================================

method_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color=color,
        label=method,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=7,
        linewidth=1.8,
    )
    for method, color in method_colors.items()
]

method_handles.append(
    Line2D(
        [0],
        [0],
        marker="x",
        color="black",
        label="Baseline ($N=1$)",
        markersize=8,
        linewidth=0,
    )
)

view_handles = []
for N in N_values:
    view_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color=method_colors["Geo"],
            markerfacecolor=shade_color(method_colors["Geo"], N),
            markeredgecolor=method_colors["Geo"],
            markersize=7,
            linewidth=0,
            label=f"$N={N}$",
        )
    )

os.makedirs("output", exist_ok=True)

# ============================================================
# COMBINED 1x3 FIGURE WITH INDEPENDENT Y-AXES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)

for idx, (ax, clf) in enumerate(zip(axes, classifiers)):

    ax.scatter(
        0,
        baseline[clf],
        marker="x",
        s=75,
        linewidths=2,
        color="black",
        zorder=6,
    )

    for method, method_data in data.items():
        base_color = method_colors[method]
        values = method_data[clf]
        xpos = method_offsets[method]

        valid_y = [v for v in values if v is not None]
        valid_x = [xpos] * len(valid_y)

        ax.plot(
            valid_x,
            valid_y,
            color=base_color,
            linewidth=1.4,
            alpha=0.55,
            zorder=2,
        )

        for N, value in zip(N_values, values):
            if value is not None:
                ax.scatter(
                    xpos,
                    value,
                    s=58,
                    color=shade_color(base_color, N),
                    edgecolor=base_color,
                    linewidth=1.0,
                    zorder=4,
                )

    ax.set_xticks([method_offsets[m] for m in method_colors])
    ax.set_xticklabels(
        list(method_colors.keys()), fontsize=9.5, rotation=30, ha="right"
    )
    ax.set_xlim(-0.25, 0.25)
    ax.set_title(clf, fontsize=13, fontweight="bold", pad=10)

    all_clf_values = [baseline[clf]]
    for m in data.values():
        all_clf_values.extend([v for v in m[clf] if v is not None])
    y_min, y_max = min(all_clf_values), max(all_clf_values)
    ax.set_ylim(y_min - 1.0, y_max + 1.0)

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if idx == 0:
        ax.set_ylabel("Balanced Accuracy (%)", fontsize=11, labelpad=8)

fig.legend(
    handles=method_handles,
    title="Method",
    loc="upper center",
    bbox_to_anchor=(0.27, -0.05),
    ncol=5,
    frameon=True,
    fontsize=8.5,
    title_fontsize=9.5,
)

fig.legend(
    handles=view_handles,
    title="Number of Views",
    loc="upper center",
    bbox_to_anchor=(0.67, -0.05),
    ncol=4,
    frameon=True,
    fontsize=8.5,
    title_fontsize=9.5,
)

plt.subplots_adjust(right=0.86, wspace=0.3)

output_path = "output/nviews_tta_combined.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved to: {output_path}")

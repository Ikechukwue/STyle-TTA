import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import os

# ============================================================
# DATA
# ============================================================

classifiers = ["CNN", "ViT", "FM"]

# Representative baseline (N=1)
baseline = {
    "CNN": 40.60,
    "ViT": 60.41,
    "FM": 69.04,
}

# Number of augmented views
N_values = [2, 4, 8, 16, 32, 64]

# Balanced accuracy (%)
data = {
    "AdaIN": {
        "CNN": [41.60, 42.15, 40.96, 42.09, 42.48, 42.34],
        "ViT": [59.07, 58.89, 58.51, 59.28, 59.25, 59.31],
        "FM": [69.10, 68.66, 66.78, 65.76, 65.32, 65.02],
    },
    "StyleID": {
        "CNN": [46.31, 48.68, 49.84, 50.32, None, None],
        "ViT": [61.94, 62.97, 63.56, 63.88, None, None],
        "FM": [69.18, 68.50, 67.73, 67.23, None, None],
    },
    "Geometric": {
        "CNN": [41.69, 42.86, 43.38, 43.67, 43.74, 44.07],
        "ViT": [60.96, 61.33, 62.23, 62.47, 62.45, 62.45],
        "FM": [69.00, 69.52, 70.25, 70.41, 70.44, 70.58],
    },
}

# ============================================================
# COLORS
# ============================================================

method_colors = {
    "AdaIN": "#d62728",       # red
    "Geometric": "#1f77b4",   # blue
    "StyleID": "#2ca02c",     # green
}

# ============================================================
# SHADE FUNCTION
# ============================================================

def shade_color(color, N):
    """Fewer views = lighter shade; more views = darker shade."""
    t = (np.log2(N) - np.log2(2)) / (np.log2(64) - np.log2(2))
    rgba = np.array(to_rgba(color))
    strength = 0.35 + 0.65 * t
    rgba[:3] = (1 - strength) * np.ones(3) + strength * rgba[:3]
    return rgba

# ============================================================
# FIGURE — COMPACT LAYOUT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.8))

# Compact family positions
x = np.array([0.0, 0.9, 1.8])

# Small offsets to separate the three methods
method_offsets = {
    "AdaIN": -0.10,
    "StyleID": 0.00,
    "Geometric": 0.10,
}

# ============================================================
# BASELINES
# ============================================================

ax.scatter(
    x,
    [baseline[c] for c in classifiers],
    marker="x",
    s=75,
    linewidths=2,
    color="black",
    zorder=6,
)

# ============================================================
# METHODS
# ============================================================

for method, method_data in data.items():

    base_color = method_colors[method]

    for classifier_idx, classifier in enumerate(classifiers):

        values = method_data[classifier]

        valid_x = []
        valid_y = []

        for N, value in zip(N_values, values):
            if value is not None:
                xpos = x[classifier_idx] + method_offsets[method]
                valid_x.append(xpos)
                valid_y.append(value)

        # Connect values across N
        ax.plot(
            valid_x,
            valid_y,
            color=base_color,
            linewidth=1.4,
            alpha=0.55,
            zorder=2,
        )

        # Individual N points
        for N, value in zip(N_values, values):

            if value is None:
                continue

            xpos = x[classifier_idx] + method_offsets[method]

            ax.scatter(
                xpos,
                value,
                s=58,
                color=shade_color(base_color, N),
                edgecolor=base_color,
                linewidth=1.0,
                zorder=4,
            )

# ============================================================
# AXES
# ============================================================

ax.set_xticks(x)
ax.set_xticklabels(classifiers, fontsize=11)

ax.set_xlabel(
    "Backbone Family",
    fontsize=12,
    labelpad=8,
)

ax.set_ylabel(
    "Balanced Accuracy (%)",
    fontsize=12,
    labelpad=8,
)

# Keep the compact x-range
ax.set_xlim(-0.35, 2.15)

# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Impact of Augmented View Count ($N$) on TTA Performance",
    fontsize=15,
    fontweight="bold",
    pad=12,
)

# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.35,
)

ax.set_axisbelow(True)

# ============================================================
# Y LIMIT
# ============================================================

all_values = []

for method in data.values():
    for classifier_values in method.values():
        all_values.extend(
            v for v in classifier_values if v is not None
        )

all_values.extend(baseline.values())

y_min = min(all_values)
y_max = max(all_values)

ax.set_ylim(y_min - 2, y_max + 2)

# ============================================================
# METHOD LEGEND
# ============================================================

method_handles = [
    Line2D(
        [0], [0],
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
        [0], [0],
        marker="x",
        color="black",
        label="Baseline ($N=1$)",
        markersize=8,
        linewidth=0,
    )
)

# ============================================================
# VIEW COUNT LEGEND
# ============================================================

view_handles = []

for N in N_values:
    view_handles.append(
        Line2D(
            [0], [0],
            marker="o",
            color=method_colors["AdaIN"],
            markerfacecolor=shade_color(
                method_colors["AdaIN"], N
            ),
            markeredgecolor=method_colors["AdaIN"],
            markersize=7,
            linewidth=0,
            label=f"$N={N}$",
        )
    )

# ============================================================
# COMPACT TWO-PART LEGEND
# ============================================================

legend1 = ax.legend(
    handles=method_handles,
    title="Method",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.00),
    frameon=True,
    fontsize=9,
    title_fontsize=10,
)

ax.add_artist(legend1)

ax.legend(
    handles=view_handles,
    title="Number of Views",
    loc="upper left",
    bbox_to_anchor=(1.01, 0.58),
    frameon=True,
    fontsize=9,
    title_fontsize=10,
)

# ============================================================
# CLEAN SPINES
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ============================================================
# LAYOUT + SAVE
# ============================================================

plt.tight_layout()

output_path = "output/nviews_tta_point_graph_compact.png"
os.makedirs("output", exist_ok=True)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved to: {output_path}")

plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

# ============================================================
# DATA
# ============================================================

classifiers = [
    "ResNet-18",
    "DenseNet-121",
    "ViT-B/16 (224)",
    "Swin-B (224)",
    "CLIP ViT-B/16",
    "DINOv2 ViT-B/14",
]

# Baseline (N=1)
baseline = {
    "ResNet-18": 35.44,
    "DenseNet-121": 40.60,
    "ViT-B/16 (224)": 47.89,
    "Swin-B (224)": 60.41,
    "CLIP ViT-B/16": 64.23,
    "DINOv2 ViT-B/14": 69.04,
}

# Number of views
N_values = [2, 4, 8, 16, 32, 64]

# Balanced accuracy values
data = {
    "AdaIN": {
        "ResNet-18":       [37.09, 38.27, 37.97, 39.01, 39.37, 39.46],
        "DenseNet-121":    [41.60, 42.15, 40.96, 42.09, 42.48, 42.34],
        "ViT-B/16 (224)":  [49.55, 50.50, 50.45, 51.84, 52.45, 52.95],
        "Swin-B (224)":    [59.07, 58.89, 58.51, 59.28, 59.25, 59.31],
        "CLIP ViT-B/16":   [64.91, 64.86, 61.69, 60.84, 60.47, 59.93],
        "DINOv2 ViT-B/14": [69.10, 68.66, 66.78, 65.76, 65.32, 65.02],
    },

    "StyleID": {
        "ResNet-18":       [41.91, 44.93, 46.28, 46.80, None, None],
        "DenseNet-121":    [46.31, 48.68, 49.84, 50.32, None, None],
        "ViT-B/16 (224)":  [52.95, 55.40, 56.75, 57.82, None, None],
        "Swin-B (224)":    [61.94, 62.97, 63.56, 63.88, None, None],
        "CLIP ViT-B/16":   [65.27, 64.35, 63.52, 62.78, None, None],
        "DINOv2 ViT-B/14": [69.18, 68.50, 67.73, 67.23, None, None],
    },

    "Geometric": {
        "ResNet-18":       [37.16, 38.01, 39.11, 39.26, 39.38, 39.46],
        "DenseNet-121":    [41.69, 42.86, 43.38, 43.67, 43.74, 44.07],
        "ViT-B/16 (224)":  [49.57, 51.07, 52.00, 52.43, 52.66, 52.80],
        "Swin-B (224)":    [60.96, 61.33, 62.23, 62.47, 62.45, 62.45],
        "CLIP ViT-B/16":   [64.65, 65.17, 65.44, 65.71, 65.72, 65.64],
        "DINOv2 ViT-B/14": [69.00, 69.52, 70.25, 70.41, 70.44, 70.58],
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
    """
    Fewer views = lighter shade
    More views = darker shade.
    """
    # Normalize N between 2 and 64
    t = (np.log2(N) - np.log2(2)) / (
        np.log2(64) - np.log2(2)
    )

    # Mix color with white
    rgba = np.array(to_rgba(color))

    # 0.35 = fairly light
    # 1.00 = full color
    strength = 0.35 + 0.65 * t

    rgba[:3] = (
        1 - strength
    ) * np.ones(3) + strength * rgba[:3]

    return rgba


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(15, 8))


# ============================================================
# PLOT BASELINES
# ============================================================

x = np.arange(len(classifiers))

ax.scatter(
    x,
    [baseline[c] for c in classifiers],
    marker="x",
    s=90,
    linewidths=2,
    color="black",
    zorder=5,
)


# ============================================================
# PLOT METHODS
# ============================================================

# Small horizontal offsets so the three methods
# don't sit exactly on top of one another.
method_offsets = {
    "AdaIN": -0.13,
    "StyleID": 0.00,
    "Geometric": 0.13,
}


for method, method_data in data.items():

    base_color = method_colors[method]

    for classifier_idx, classifier in enumerate(classifiers):

        values = method_data[classifier]

        valid_x = []
        valid_y = []

        for N, value in zip(N_values, values):

            if value is None:
                continue

            valid_x.append(
                classifier_idx + method_offsets[method]
            )

            valid_y.append(value)

        # ----------------------------------------------------
        # Connecting line
        # ----------------------------------------------------

        ax.plot(
            valid_x,
            valid_y,
            color=base_color,
            linewidth=1.5,
            alpha=0.55,
            zorder=2,
        )

        # ----------------------------------------------------
        # Individual points
        # ----------------------------------------------------

        for N, value in zip(N_values, values):

            if value is None:
                continue

            xpos = (
                classifier_idx
                + method_offsets[method]
            )

            ax.scatter(
                xpos,
                value,
                s=65,
                color=shade_color(
                    base_color,
                    N
                ),
                edgecolor=base_color,
                linewidth=1.2,
                zorder=4,
            )


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(
    classifiers,
    fontsize=11,
)

ax.set_xlabel(
    "Classifier",
    fontsize=13,
    labelpad=12,
)

ax.set_ylabel(
    "Balanced Accuracy (%)",
    fontsize=13,
    labelpad=12,
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Impact of Augmented View Count ($N$) on TTA Performance",
    fontsize=17,
    fontweight="bold",
    pad=18,
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
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
            v for v in classifier_values
            if v is not None
        )

all_values.extend(baseline.values())

y_min = min(all_values)
y_max = max(all_values)

ax.set_ylim(
    y_min - 2,
    y_max + 2,
)


# ============================================================
# METHOD LEGEND
# ============================================================

method_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color=color,
        label=method,
        markerfacecolor=color,
        markersize=8,
        linewidth=2,
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
        markersize=9,
        linewidth=0,
    )
)


# ============================================================
# VIEW COUNT LEGEND
# ============================================================

view_handles = []

for N in N_values:

    # Use AdaIN red only as an example
    # to communicate the shade scale.
    view_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color=method_colors["AdaIN"],
            markerfacecolor=shade_color(
                method_colors["AdaIN"],
                N
            ),
            markeredgecolor=method_colors["AdaIN"],
            markersize=8,
            linewidth=0,
            label=f"$N={N}$",
        )
    )


# ============================================================
# LEGENDS
# ============================================================

legend1 = ax.legend(
    handles=method_handles,
    title="Method",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.00),
    frameon=True,
)

ax.add_artist(legend1)

ax.legend(
    handles=view_handles,
    title="Number of Views",
    loc="upper left",
    bbox_to_anchor=(1.01, 0.68),
    frameon=True,
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = "output/nviews_tta_point_graph.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved to: {output_path}")


# ============================================================
# SHOW
# ============================================================



import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

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
    "Geo": {
        "CNN": [41.69, 42.86, 43.38, 43.67, 43.74, 44.07],
        "ViT": [60.96, 61.33, 62.23, 62.47, 62.45, 62.45],
        "FM": [69.00, 69.52, 70.25, 70.41, 70.44, 70.58],
    },
}

# ============================================================
# COLORS & OFFSETS
# ============================================================

method_colors = {
    "AdaIN": "#d62728",
    "Geo": "#1f77b4",
    "StyleID": "#2ca02c",
}

method_offsets = {
    "AdaIN": -0.10,
    "StyleID": 0.00,
    "Geo": 0.10,
}

# ============================================================
# SHADE FUNCTION
# ============================================================


def shade_color(color, N):
  t = (np.log2(N) - np.log2(2)) / (np.log2(64) - np.log2(2))
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
          color=method_colors["AdaIN"],
          markerfacecolor=shade_color(method_colors["AdaIN"], N),
          markeredgecolor=method_colors["AdaIN"],
          markersize=7,
          linewidth=0,
          label=f"$N={N}$",
      )
  )

os.makedirs("output", exist_ok=True)

# ============================================================
# COMBINED 1x3 FIGURE WITH INDEPENDENT Y-AXES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

for idx, (ax, clf) in enumerate(zip(axes, classifiers)):

  # Plot baseline
  ax.scatter(
      0,
      baseline[clf],
      marker="x",
      s=75,
      linewidths=2,
      color="black",
      zorder=6,
  )

  # Plot methods
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

  # Subplot layout
  ax.set_xticks([method_offsets[m] for m in method_colors])
  ax.set_xticklabels(list(method_colors.keys()), fontsize=10)
  ax.set_xlim(-0.25, 0.25)
  ax.set_title(clf, fontsize=13, fontweight="bold", pad=10)

  # Dynamic y-axis range per backbone family
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

# Unified shared legends
fig.legend(
    handles=method_handles,
    title="Method",
    loc="upper center",
    bbox_to_anchor=(0.27, -0.02),
    ncol=4,
    frameon=True,
    fontsize=8.5,
    title_fontsize=9.5,
)

fig.legend(
    handles=view_handles,
    title="Number of Views",
    loc="upper center",
    bbox_to_anchor=(0.67, -0.02),
    ncol=6,
    frameon=True,
    fontsize=8.5,
    title_fontsize=9.5,
)

plt.subplots_adjust(right=0.86, wspace=0.3)

output_path = "output/nviews_tta_combined.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved to: {output_path}")

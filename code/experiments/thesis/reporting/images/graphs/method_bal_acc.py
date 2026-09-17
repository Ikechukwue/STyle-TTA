import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# Balanced Accuracy by Classifier Group
# CNN  -> DenseNet-121
# ViT  -> Swin-B (224)
# FM   -> DINOv2 ViT-B/14
# ---------------------------------------------------------

# Classifier groups
groups = ["CNN", "ViT", "FM"]

# Balanced accuracy (%) from the table
adain = [42.34, 59.31, 65.02]
styleid = [50.32, 63.88, 67.23]
geometric = [44.07, 62.45, 70.58]

# Backbone labels corresponding to each classifier group
backbones = [
    "DenseNet-121",
    "Swin-B (224)",
    "DINOv2 ViT-B/14"
]

# X positions
x = np.arange(len(groups))

# Bar width
width = 0.23

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

# Bars
bars_adain = ax.bar(
    x - width,
    adain,
    width,
    label="AdaIN",
    color="#d62728"
)

bars_styleid = ax.bar(
    x,
    styleid,
    width,
    label="StyleID",
    color="#2ca02c"
)

bars_geometric = ax.bar(
    x + width,
    geometric,
    width,
    label="Geometric",
    color="#1f77b4"
)

# ---------------------------------------------------------
# Add values above bars
# ---------------------------------------------------------

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10
        )

add_labels(bars_adain)
add_labels(bars_styleid)
add_labels(bars_geometric)

# ---------------------------------------------------------
# Axis labels and title
# ---------------------------------------------------------

ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
ax.set_xlabel("Classifier Group", fontsize=12)

ax.set_title(
    "Balanced Accuracy by Classifier Group",
    fontsize=14,
    fontweight="bold"
)

# X-axis
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)

# Y-axis
ax.set_ylim(0, 80)

# Legend
ax.legend(
    title="TTA Strategy",
    fontsize=10,
    title_fontsize=10
)

# Grid
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

# Keep grid behind bars
ax.set_axisbelow(True)

# ---------------------------------------------------------
# Add backbone names below classifier groups
# ---------------------------------------------------------

for i, backbone in enumerate(backbones):
    ax.text(
        i,
        -6,
        backbone,
        ha="center",
        va="top",
        fontsize=9
    )

# Give some room for the backbone labels
plt.subplots_adjust(bottom=0.18)

# Save high-resolution figure
plt.savefig(
    "output/images/balanced_accuracy_classifier_groups.png",
    dpi=300,
    bbox_inches="tight"
)


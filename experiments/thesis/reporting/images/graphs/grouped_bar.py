import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = [
    ["EuroSAT", "CNN", 18.84, -6.58],
    ["EuroSAT", "Vision Transformer", 0.34, 3.50],
    ["EuroSAT", "Foundation Model", -3.17, -0.75],
    ["Camelyon", "CNN", 12.56, 16.49],
    ["Camelyon", "Vision Transformer", 2.45, 0.06],
    ["Camelyon", "Foundation Model", 2.01, 0.84],
    ["MIDOG", "CNN", 1.05, 0.99],
    ["MIDOG", "Vision Transformer", -2.47, -1.49],
    ["MIDOG", "Foundation Model", 0.77, -0.78],
    ["EpiStr", "CNN", 17.24, 26.82],
    ["EpiStr", "Vision Transformer", 0.54, -2.39],
    ["EpiStr", "Foundation Model", 0.75, 8.65],
]

df = pd.DataFrame(
    data, columns=["Dataset", "Backbone", "Delta_STyle-TTA", "Delta_Geometric"]
)

backbones = ["CNN", "Vision Transformer", "Foundation Model"]
backbone_titles = {"CNN": "CNN", "Vision Transformer": "ViT", "Foundation Model": "FM"}
datasets = ["EuroSAT", "Camelyon", "MIDOG", "EpiStr"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
x = np.arange(len(datasets))
width = 0.35

for idx, bb in enumerate(backbones):
  ax = axes[idx]
  sub_df = df[df["Backbone"] == bb].set_index("Dataset").loc[datasets]

  ax.bar(
      x - width / 2,
      sub_df["Delta_STyle-TTA"],
      width,
      label="STyle-TTA",
      color="#2ca02c",
  )
  ax.bar(
      x + width / 2,
      sub_df["Delta_Geometric"],
      width,
      label="Geometric",
      color="#1f77b4",
  )

  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
  ax.set_title(
      f"Backbone: {backbone_titles[bb]}", fontsize=12, fontweight="bold"
  )
  ax.set_xticks(x)
  ax.set_xticklabels(datasets, rotation=15, ha="right")
  ax.grid(axis="y", linestyle=":", alpha=0.6)

  if idx == 0:
    ax.set_ylabel(
        "$\Delta$ Balanced Accuracy (%)", fontsize=11, fontweight="bold"
    )
    ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("output/images/bar_plot.png", dpi=300)
plt.close()

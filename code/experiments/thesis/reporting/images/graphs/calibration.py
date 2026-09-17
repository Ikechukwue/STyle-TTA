import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

def create_ece_plot():
    data = [
        {"Backbone": "ResNet-18", "Family": "CNN", "Strategy": "AdaIN", "Accuracy": 39.46, "AUC": 0.78},
        {"Backbone": "ResNet-18", "Family": "CNN", "Strategy": "STyle-TTA", "Accuracy": 46.80, "AUC": 0.94},
        {"Backbone": "ResNet-18", "Family": "CNN", "Strategy": "Geometric", "Accuracy": 39.46, "AUC": 0.92},
        {"Backbone": "DenseNet-121", "Family": "CNN", "Strategy": "AdaIN", "Accuracy": 42.34, "AUC": 0.80},
        {"Backbone": "DenseNet-121", "Family": "CNN", "Strategy": "STyle-TTA", "Accuracy": 50.32, "AUC": 0.95},
        {"Backbone": "DenseNet-121", "Family": "CNN", "Strategy": "Geometric", "Accuracy": 44.07, "AUC": 0.93},
        {"Backbone": "ViT-B/16 (224)", "Family": "ViT", "Strategy": "AdaIN", "Accuracy": 52.95, "AUC": 0.85},
        {"Backbone": "ViT-B/16 (224)", "Family": "ViT", "Strategy": "STyle-TTA", "Accuracy": 57.82, "AUC": 0.97},
        {"Backbone": "ViT-B/16 (224)", "Family": "ViT", "Strategy": "Geometric", "Accuracy": 52.80, "AUC": 0.95},
        {"Backbone": "Swin-B (224)", "Family": "ViT", "Strategy": "AdaIN", "Accuracy": 59.31, "AUC": 0.87},
        {"Backbone": "Swin-B (224)", "Family": "ViT", "Strategy": "STyle-TTA", "Accuracy": 63.88, "AUC": 0.98},
        {"Backbone": "Swin-B (224)", "Family": "ViT", "Strategy": "Geometric", "Accuracy": 62.45, "AUC": 0.97},
        {"Backbone": "CLIP ViT-B/16", "Family": "FM", "Strategy": "AdaIN", "Accuracy": 59.93, "AUC": 0.90},
        {"Backbone": "CLIP ViT-B/16", "Family": "FM", "Strategy": "STyle-TTA", "Accuracy": 62.78, "AUC": 0.98},
        {"Backbone": "CLIP ViT-B/16", "Family": "FM", "Strategy": "Geometric", "Accuracy": 65.64, "AUC": 0.99},
        {"Backbone": "DINOv2 ViT-B/14", "Family": "FM", "Strategy": "AdaIN", "Accuracy": 65.02, "AUC": 0.89},
        {"Backbone": "DINOv2 ViT-B/14", "Family": "FM", "Strategy": "STyle-TTA", "Accuracy": 67.23, "AUC": 0.98},
        {"Backbone": "DINOv2 ViT-B/14", "Family": "FM", "Strategy": "Geometric", "Accuracy": 70.58, "AUC": 0.98},
    ]

    df = pd.DataFrame(data)
    df = df[df["Strategy"] != "AdaIN"]
    plt.figure(figsize=(9, 6))

    markers = {'CNN': 'o', 'ViT': 's', 'FM': '^'}

    sns.scatterplot(
        data=df,
        x="AUC",
        y="Accuracy",
        hue="Strategy",
        style="Family",
        markers=markers,
        s=120,
        alpha=0.9
    )

    plt.title("AUC vs. Balanced Accuracy by Strategy and Backbone Family")
    plt.xlabel("Area Under ROC Curve (AUC) ↑")
    plt.ylabel("Balanced Accuracy (%) ↑")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig("output/images/auc_plot.png", dpi=300)
    plt.close()


def create_auc_plot():
    data = [
        {"Backbone": "ResNet-18", "Family": "CNN", "Strategy": "AdaIN", "Accuracy": 39.46, "ECE": 0.29},
        {"Backbone": "ResNet-18", "Family": "CNN", "Strategy": "STyle-TTA", "Accuracy": 46.80, "ECE": 0.05},
        {"Backbone": "ResNet-18", "Family": "CNN", "Strategy": "Geometric", "Accuracy": 39.46, "ECE": 0.04},
        {"Backbone": "DenseNet-121", "Family": "CNN", "Strategy": "AdaIN", "Accuracy": 42.34, "ECE": 0.27},
        {"Backbone": "DenseNet-121", "Family": "CNN", "Strategy": "STyle-TTA", "Accuracy": 50.32, "ECE": 0.09},
        {"Backbone": "DenseNet-121", "Family": "CNN", "Strategy": "Geometric", "Accuracy": 44.07, "ECE": 0.08},
        {"Backbone": "ViT-B/16 (224)", "Family": "ViT", "Strategy": "AdaIN", "Accuracy": 52.95, "ECE": 0.19},
        {"Backbone": "ViT-B/16 (224)", "Family": "ViT", "Strategy": "STyle-TTA", "Accuracy": 57.82, "ECE": 0.13},
        {"Backbone": "ViT-B/16 (224)", "Family": "ViT", "Strategy": "Geometric", "Accuracy": 52.80, "ECE": 0.11},
        {"Backbone": "Swin-B (224)", "Family": "ViT", "Strategy": "AdaIN", "Accuracy": 59.31, "ECE": 0.18},
        {"Backbone": "Swin-B (224)", "Family": "ViT", "Strategy": "STyle-TTA", "Accuracy": 63.88, "ECE": 0.13},
        {"Backbone": "Swin-B (224)", "Family": "ViT", "Strategy": "Geometric", "Accuracy": 62.45, "ECE": 0.19},
        {"Backbone": "CLIP ViT-B/16", "Family": "FM", "Strategy": "AdaIN", "Accuracy": 59.93, "ECE": 0.17},
        {"Backbone": "CLIP ViT-B/16", "Family": "FM", "Strategy": "STyle-TTA", "Accuracy": 62.78, "ECE": 0.07},
        {"Backbone": "CLIP ViT-B/16", "Family": "FM", "Strategy": "Geometric", "Accuracy": 65.64, "ECE": 0.13},
        {"Backbone": "DINOv2 ViT-B/14", "Family": "FM", "Strategy": "AdaIN", "Accuracy": 65.02, "ECE": 0.19},
        {"Backbone": "DINOv2 ViT-B/14", "Family": "FM", "Strategy": "STyle-TTA", "Accuracy": 67.23, "ECE": 0.05},
        {"Backbone": "DINOv2 ViT-B/14", "Family": "FM", "Strategy": "Geometric", "Accuracy": 70.58, "ECE": 0.09},
    ]

    df = pd.DataFrame(data)
    df = df[df["Strategy"] != "AdaIN"]
    plt.figure(figsize=(9, 6))

    markers = {'CNN': 'o', 'ViT': 's', 'FM': '^'}

    sns.scatterplot(
        data=df,
        x="ECE",
        y="Accuracy",
        hue="Strategy",
        style="Family",
        markers=markers,
        s=120,
        alpha=0.9
    )

    plt.title("ECE vs. Balanced Accuracy by Strategy and Backbone Family")
    plt.xlabel("Expected Calibration Error (ECE) ↓")
    plt.ylabel("Balanced Accuracy (%) ↑")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig("output/images/ece_plot.png", dpi=300)
    plt.close()


def united():
    data = [
        ["ResNet-18", "AdaIN", 0.29, 0.78],
        ["ResNet-18", "STyle-TTA", 0.05, 0.94],
        ["ResNet-18", "Geometric", 0.04, 0.92],
        ["DenseNet-121", "AdaIN", 0.27, 0.80],
        ["DenseNet-121", "STyle-TTA", 0.09, 0.95],
        ["DenseNet-121", "Geometric", 0.08, 0.93],
        ["ViT-B/16", "AdaIN", 0.19, 0.85],
        ["ViT-B/16", "STyle-TTA", 0.13, 0.97],
        ["ViT-B/16", "Geometric", 0.11, 0.95],
        ["Swin-B", "AdaIN", 0.18, 0.87],
        ["Swin-B", "STyle-TTA", 0.13, 0.98],
        ["Swin-B", "Geometric", 0.19, 0.97],
        ["CLIP", "AdaIN", 0.17, 0.90],
        ["CLIP", "STyle-TTA", 0.07, 0.98],
        ["CLIP", "Geometric", 0.13, 0.99],
        ["DINOv2", "AdaIN", 0.19, 0.89],
        ["DINOv2", "STyle-TTA", 0.05, 0.98],
        ["DINOv2", "Geometric", 0.09, 0.98],
    ]

    base_metrics = {
        "ResNet-18": {"auc": 0.9020, "ece": 0.1357},
        "DenseNet-121": {"auc": 0.9137, "ece": 0.0406},
        "ViT-B/16": {"auc": 0.9331, "ece": 0.0154},
        "Swin-B": {"auc": 0.9627, "ece": 0.0960},
        "CLIP": {"auc": 0.9860, "ece": 0.1124},
        "DINOv2": {"auc": 0.9841, "ece": 0.0268},
    }

    df = pd.DataFrame(data, columns=["Backbone", "Strategy", "ECE", "AUC"])
    df = df[df["Strategy"] != "AdaIN"]
    backbones = df["Backbone"].unique()
    strategies = ["STyle-TTA", "Geometric"]

    x = np.arange(len(backbones))
    width = 0.25
    colors = {   "Geometric": "#1f77b4",   # blue
        "STyle-TTA": "#2ca02c"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for i, strat in enumerate(strategies):
        strat_df = (
            df[df["Strategy"] == strat]
            .set_index("Backbone")
            .loc[backbones]
            .reset_index()
        )
        offset = (i - 0.5) * width
        ax1.bar(
            x + offset, strat_df["AUC"], width, label=strat, color=colors[strat]
        )
        ax2.bar(
            x + offset, strat_df["ECE"], width, label=strat, color=colors[strat]
        )

    base_added = False
    for idx, bb in enumerate(backbones):
        if bb in base_metrics:
            auc_val = base_metrics[bb]["auc"]
            ece_val = base_metrics[bb]["ece"]
            xmin = idx - width
            xmax = idx + width
            label = "Baseline" if not base_added else ""

            ax1.hlines(
                y=auc_val,
                xmin=xmin,
                xmax=xmax,
                colors="black",
                linestyles="dotted",
                linewidth=1.8,
                label=label,
            )
            ax2.hlines(
                y=ece_val,
                xmin=xmin,
                xmax=xmax,
                colors="black",
                linestyles="dotted",
                linewidth=1.8,
                label=label,
            )
            base_added = True

    ax1.set_title(r"(a) Ranking Performance (AUC $\uparrow$)")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.75, 1.03)
    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(title="Strategy", loc="lower right")

    ax2.set_title(r"(b) Expected Calibration Error (ECE $\downarrow$)")
    ax2.set_ylabel("ECE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(backbones, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(title="Strategy", loc="upper right")

    plt.tight_layout()
    plt.savefig("output/images/plot.png", dpi=300)
    plt.close()



def small():
    labels = [
        "Camelyon\n(CNN)",
        "Camelyon\n(ViT)",
        "Camelyon\n(FM)",
        "EpiStr\n(CNN)",
        "EpiStr\n(ViT)",
        "EpiStr\n(FM)",
    ]

    # Panel (a): Balanced Accuracy (%)
    style_acc = [80.22, 95.21, 94.98, 67.36, 70.41, 66.45]
    geo_acc = [84.15, 92.82, 93.81, 76.94, 67.48, 74.35]

    # Panel (b): Expected Calibration Error (ECE)
    style_ece = [0.14, 0.02, 0.02, 0.18, 0.12, 0.20]
    geo_ece = [0.10, 0.07, 0.04, 0.08, 0.21, 0.17]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel (a): Balanced Accuracy ---
    rects1 = ax1.bar(
        x - width / 2,
        style_acc,
        width,
        label="STyle-TTA ($N=16$)",
        color="#2ca02c",
    )
    rects2 = ax1.bar(
        x + width / 2,
        geo_acc,
        width,
        label="Geometric ($N=64$)",
        color="#1f77b4",
    )

    ax1.set_ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Balanced Accuracy Context", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Group separation lines (between Camelyon and EpiStr)
    ax1.axvline(2.5, color="gray", linestyle="--", alpha=0.5)

    # --- Panel (b): Expected Calibration Error ---
    rects3 = ax2.bar(
        x - width / 2,
        style_ece,
        width,
        label="STyle-TTA ($N=16$)",
        color="#2ca02c",
    )
    rects4 = ax2.bar(
        x + width / 2,
        geo_ece,
        width,
        label="Geometric ($N=64$)",
        color="#1f77b4",
    )

    ax2.set_ylabel("ECE ↓", fontsize=11, fontweight="bold")
    ax2.set_title(
        "(b) Expected Calibration Error (ECE)", fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
  

    # Group separation lines
    ax2.axvline(2.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("calibration_two_panel_figure.png", dpi=300, bbox_inches="tight")
    plt.close()

def imagenet_small():

    # Selected 3 models: DenseNet-121 (CNN), Swin-B (ViT), DINOv2 (FM)
    labels = ["DenseNet-121\n(CNN)", "Swin-B\n(ViT)", "DINOv2\n(FM)"]

    # Panel (a): Balanced Accuracy (%) on ImageNet-R
    style_acc = [50.32, 63.88, 67.23]
    geo_acc = [44.07, 62.45, 70.58]

    # Panel (b): Expected Calibration Error (ECE)
    style_ece = [0.09, 0.13, 0.05]
    geo_ece = [0.08, 0.19, 0.09]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Panel (a): Balanced Accuracy ---
    ax1.bar(
        x - width / 2,
        style_acc,
        width,
        label="STyle-TTA ($N=16$)",
        color="#2ca02c",
    )
    ax1.bar(
        x + width / 2,
        geo_acc,
        width,
        label="Geometric ($N=64$)",
        color="#1f77b4",
    )

    ax1.set_ylabel("Balanced Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "(a) Balanced Accuracy on ImageNet-R", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(30, 80)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # --- Panel (b): Expected Calibration Error ---
    ax2.bar(
        x - width / 2,
        style_ece,
        width,
        label="StyleID ($N=16$)",
        color="#2ca02c",
    )
    ax2.bar(
        x + width / 2,
        geo_ece,
        width,
        label="Geometric ($N=64$)",
        color="#1f77b4",
    )

    ax2.set_ylabel("ECE ↓ (Lower is better)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "(b) Expected Calibration Error (ECE)", fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 0.25)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    

    plt.tight_layout()
    plt.savefig("calibration_two_panel_figure.png", dpi=300, bbox_inches="tight")
    plt.close()
if __name__ == "__main__":
    imagenet_small()

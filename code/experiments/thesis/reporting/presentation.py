import json
import matplotlib.pyplot as plt
import numpy as np
from code.config.helpers import load_json, get_classifier_name
def plot_model_accuracies(json_string):
    data = load_json(json_string)
    models_data = data["metrics"]["imagenet"]
    models = list(models_data.keys())
    
    val_acc = []
    val_subset_acc = []
    test_r_acc = []
    
    for model in models:
        metrics = models_data[model]["none"]["71397589"]
        val_acc.append(metrics["val"]["accuracy"])
        val_subset_acc.append(metrics["val@test_r"]["accuracy"])
        test_r_acc.append(metrics["test_r"]["accuracy"])
        
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, test_r_acc, width, label='ImageNet-R', color='tab:orange', zorder=3)
    ax.bar(x, val_subset_acc, width, label='IN-1k Val subset of 200 classes', color='#6BAED6', zorder=3)
    ax.bar(x + width, val_acc, width, label='IN-1k Val', facecolor='none', edgecolor='#08519C', linestyle='--', linewidth=1.5, zorder=3)
    
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_xticks(x)
    
    display_labels = [get_classifier_name(m)[0] for m in models]
    ax.set_xticklabels(display_labels, rotation=15, ha='right')
    
    ax.legend(title="Datasets", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    
    fig.tight_layout()
    fig.savefig('model_accuracies.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
if __name__ == "__main__":
    path = "/home/stud/nemmler/retristyle/results/classifier_eval/classifier_evaluation/metrics_base.json"
    plot_model_accuracies(path)

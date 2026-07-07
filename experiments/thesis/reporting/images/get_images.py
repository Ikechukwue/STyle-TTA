import numpy as np
import random
import matplotlib.pyplot as plt
from config.helpers import get_names, load_json
from torch.utils.data import DataLoader
from experiments.data import create_dataset
from PIL import Image

def get_samples(indices, dataloader, pred1, pred2, y_true):
    images = [dataloader.dataset[i][0] for i in indices]
    w_preds = [pred1[i] for i in indices]  
    f_preds = [pred2[i] for i in indices]  
    trues = [y_true[i] for i in indices]
    return images, trues, w_preds, f_preds

def get_misclassified_indices(results: dict):
    preds = np.array([sample["y_pred"] for sample in results["predictions"]])
    y_pred_idx = np.argmax(preds, axis=-1)
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    return np.where(y_pred_idx != y_true)[0], y_pred_idx, y_true

def plot_recovery_images(images, true_labels, wrong_preds, fixed_preds,indices , split_name, filename):
    class_names = get_names(split=split_name)
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).flatten()
    
    for i in range(len(axes)):
        if i < n:
            img = np.array(images[i])
            # If black, ensure values are in [0, 1] range; if normalized, add logic to reverse here
            if img.max() > 1:
                img = img / 255.0
            axes[i].imshow(img)
            title = f"ID: {indices[i]}\nTrue: {class_names[true_labels[i]]}\nInit: {class_names[wrong_preds[i]]}\nFix: {class_names[fixed_preds[i]]}"
            axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    loader = DataLoader(create_dataset("imagenet", "./data", "test_r"), batch_size=1, shuffle=False)
    
    d1 = load_json("results/baseline/tta_inference/predictions/imagenet/test_r/densenet121_geometric_vanilla_nviews1_seed71397589.json")
    d2 = load_json("results/ablation/retristyle/tta_inference/predictions/imagenet/test_r/densenet121_retristyle_vanilla_dino_nrefs2_seed71397589.json")
    
    idx1, pred1, true1 = get_misclassified_indices(d1)
    idx2, pred2, true2 = get_misclassified_indices(d2)
    
    recovered = [i for i in idx1 if i not in idx2]
    sampled = random.sample(recovered, min(len(recovered), 5))
    
    # Corrected call: pass both prediction sets
    imgs, t_labs, w_preds, f_preds = get_samples(sampled, loader, pred1, pred2, true1)
    
    plot_recovery_images(imgs, t_labs, w_preds, f_preds, sampled, "test_r", "recovered_predictions_2.png")

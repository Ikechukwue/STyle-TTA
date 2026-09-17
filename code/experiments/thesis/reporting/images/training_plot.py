import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_path):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df['epoch'], df['train/loss'], label='Train Loss')
    axes[0].plot(df['epoch'], df['val/loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df['epoch'], df['train/accuracy'], label='Train Accuracy')
    axes[1].plot(df['epoch'], df['val/accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('output/training_metrics.png', dpi=300)
    plt.close()

csv_path = '/home/stud/nemmler/retristyle/data/models/test/camelyon17wilds/camelyon17wilds-densenet121-random_flip-random_resized_crop-seed42_metrics.csv'
plot_training_results(csv_path)

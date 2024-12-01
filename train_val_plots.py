import matplotlib.pyplot as plt

def plot_training_history(history, dpi=100):
    """Plots the training and validation metrics without saving."""
    metrics = [metric for metric in history.history.keys() if not metric.startswith('val_')]

    plt.figure(figsize=(16, 6), dpi=dpi)  # Larger figure size to accommodate side-by-side

    # Plot each metric side by side
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, len(metrics), i)  # 1 row, n columns
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel(metric.capitalize(), fontsize=10)
        plt.title(f'Training and Validation {metric.capitalize()}', fontsize=12)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True)

    plt.tight_layout()  # Adjust spacing to prevent overlap
    plt.show()

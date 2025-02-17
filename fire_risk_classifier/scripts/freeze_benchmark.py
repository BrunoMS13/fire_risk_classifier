import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_json_data(file_path):
    """Load JSON data safely and return None if file is missing."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    with open(file_path, "r") as f:
        return json.load(f)


def get_max_epochs(models_dict):
    """Find the maximum number of epochs across all models."""
    return max(max(len(data[key]) for key in data.keys()) for data in models_dict.values() if data)


def plot_learning_rates(models_dict):
    """Plot training metrics for multiple learning rate experiments in the same graphs."""
    
    # Find the max number of epochs across all models
    max_epochs = max(len(data.get("train_loss", [])) for data in models_dict.values())
    epochs = np.arange(1, max_epochs + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Helper function to safely plot truncated data
    def safe_plot(ax, x_values, y_values, label, linestyle="-"):
        if len(y_values) > 0:
            ax.plot(x_values[: len(y_values)], y_values, label=label, linestyle=linestyle)

    # Loss Plot
    for lr, data in models_dict.items():
        safe_plot(axs[0], epochs, data.get("train_loss", []), f"Train Loss (LR {lr})", "--")
        safe_plot(axs[0], epochs, data.get("val_loss", []), f"Val Loss (LR {lr})", "-")

    axs[0].set_title("Training & Validation Loss Across Learning Rates")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy Plot
    for lr, data in models_dict.items():
        safe_plot(axs[1], epochs, data.get("train_accuracy", []), f"Train Acc (LR {lr})", "--")
        safe_plot(axs[1], epochs, data.get("val_accuracy", []), f"Val Acc (LR {lr})", "-")

    axs[1].set_title("Training & Validation Accuracy Across Learning Rates")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(True)

    # F1 Score Plot
    for lr, data in models_dict.items():
        safe_plot(axs[2], epochs, data.get("train_f1_score", []), f"Train F1 (LR {lr})", "--")
        safe_plot(axs[2], epochs, data.get("val_f1_score", []), f"Val F1 (LR {lr})", "-")

    axs[2].set_title("Training & Validation F1 Score Across Learning Rates")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()
    axs[2].grid(True)

    # Ensure integer epoch labels
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


# **Define the learning rate models to compare (all on one plot)**
model = "densenet161"
learning_rate_models = {
    "1e-4": load_json_data(f"models/models_old/{model}_irg_wd5e-4_lr1e-4_patience8_metrics.json"),
    "1e-5": load_json_data(f"models/models_old/{model}_irg_wd5e-4_lr1e-5_patience8_metrics.json"),
    "5e-5": load_json_data(f"models/models_old/{model}_irg_wd5e-4_lr5e-5_patience8_metrics.json"),
    "5e-6": load_json_data(f"models/models_old/{model}_irg_wd5e-4_lr5e-6_patience8_metrics.json"),
    "1e-6": load_json_data(f"models/models_old/{model}_irg_wd5e-4_lr1e-6_patience8_metrics.json"),
    "5e-7": load_json_data(f"models/models_old/{model}_irg_wd5e-4_lr5e-7_patience8_metrics.json"),
}

# Remove any None values in case a file wasn't found
learning_rate_models = {lr: data for lr, data in learning_rate_models.items() if data}

# Call the function to plot
def main():
    #print(learning_rate_models)
    for learning_rate, data in learning_rate_models.items():
        if data:  # Ensure data is not empty
            # Find the index of the lowest validation loss
            lowest_val_loss_idx = data["val_loss"].index(min(data["val_loss"]))

            # Get values from the same epoch
            lowest_val_loss = data["val_loss"][lowest_val_loss_idx]
            corresponding_train_loss = data["train_loss"][lowest_val_loss_idx]
            corresponding_val_acc = data["val_accuracy"][lowest_val_loss_idx]
            corresponding_train_acc = data["train_accuracy"][lowest_val_loss_idx]

            print(f"Learning Rate: {learning_rate}, "
                f"Val Acc (At Min Val Loss Epoch): {corresponding_val_acc}, "
                f"Train Acc (At Min Val Loss Epoch): {corresponding_train_acc}, "
                f"Lowest Val Loss: {lowest_val_loss}, "
                f"Train Loss (At Min Val Loss Epoch): {corresponding_train_loss}")
    if learning_rate_models:
        plot_learning_rates(learning_rate_models)
    else:
        print("Could not load any JSON data. Please check file paths.")

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def truncate_to_shortest(data_dict1, data_dict2):
    """Truncate all keys in both dictionaries to match the shortest available length."""
    truncated_dict1 = {}
    truncated_dict2 = {}

    for key in data_dict1.keys():
        if key in data_dict2:
            min_length = min(len(data_dict1[key]), len(data_dict2[key]))
            truncated_dict1[key] = data_dict1[key][:min_length]
            truncated_dict2[key] = data_dict2[key][:min_length]

    return truncated_dict1, truncated_dict2


def plot_metrics(fully_unfrozen, gradual_unfreezing):
    # Truncate each dataset to its shortest length
    fully_unfrozen_trunc, gradual_unfreezing_trunc = truncate_to_shortest(fully_unfrozen, gradual_unfreezing)

    # Determine number of epochs after truncation
    epochs = np.arange(1, len(fully_unfrozen_trunc["train_loss"]) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Loss Plot
    axs[0].plot(epochs, fully_unfrozen_trunc["train_loss"], label="Train Loss (Fully Unfrozen)", linestyle="--")
    axs[0].plot(epochs, fully_unfrozen_trunc["val_loss"], label="Val Loss (Fully Unfrozen)", linestyle="--")
    axs[0].plot(epochs, gradual_unfreezing_trunc["train_loss"], label="Train Loss (Gradual Unfreezing)", linestyle="-")
    axs[0].plot(epochs, gradual_unfreezing_trunc["val_loss"], label="Val Loss (Gradual Unfreezing)", linestyle="-")
    axs[0].set_title("Training & Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy Plot
    axs[1].plot(epochs, fully_unfrozen_trunc["train_accuracy"], label="Train Acc (Fully Unfrozen)", linestyle="--")
    axs[1].plot(epochs, fully_unfrozen_trunc["val_accuracy"], label="Val Acc (Fully Unfrozen)", linestyle="--")
    axs[1].plot(epochs, gradual_unfreezing_trunc["train_accuracy"], label="Train Acc (Gradual Unfreezing)", linestyle="-")
    axs[1].plot(epochs, gradual_unfreezing_trunc["val_accuracy"], label="Val Acc (Gradual Unfreezing)", linestyle="-")
    axs[1].set_title("Training & Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(True)

    # F1 Score Plot
    axs[2].plot(epochs, fully_unfrozen_trunc["train_f1_score"], label="Train F1 (Fully Unfrozen)", linestyle="--")
    axs[2].plot(epochs, fully_unfrozen_trunc["val_f1_score"], label="Val F1 (Fully Unfrozen)", linestyle="--")
    axs[2].plot(epochs, gradual_unfreezing_trunc["train_f1_score"], label="Train F1 (Gradual Unfreezing)", linestyle="-")
    axs[2].plot(epochs, gradual_unfreezing_trunc["val_f1_score"], label="Val F1 (Gradual Unfreezing)", linestyle="-")
    axs[2].set_title("Training & Validation F1 Score")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("F1 Score")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# Function to load JSON data with error handling
def load_json_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data

# Change these filenames to your actual files
fully_unfrozen_file = "models/models/r101_irg_wd5e-4_lr5e-4_unfreezeGradual_metrics.json"  
gradual_unfreezing_file = "models/models/r101_irg_wd5e-4_lr5e-4_unfreezeNothing_metrics.json"  

# Load the JSON files
fully_unfrozen = load_json_data(fully_unfrozen_file)
gradual_unfreezing = load_json_data(gradual_unfreezing_file)

# Call the function to plot
def main():
    if fully_unfrozen and gradual_unfreezing:
        plot_metrics(fully_unfrozen, gradual_unfreezing)
    else:
        print("Could not load JSON data. Please check file paths.")

main()

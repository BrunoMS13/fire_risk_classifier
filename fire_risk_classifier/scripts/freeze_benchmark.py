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
    return max(
        max(len(data[key]) for key in data.keys())
        for data in models_dict.values()
        if data
    )


def plot_learning_rates(models_dict):
    """Plot training and validation metrics separately for multiple learning rate experiments."""

    # Find the max number of epochs across all models
    max_epochs = max(len(data.get("train_loss", [])) for data in models_dict.values())
    epochs = np.arange(1, max_epochs + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Helper function to safely plot truncated data
    def safe_plot(ax, x_values, y_values, label, linestyle="-", color=None):
        if len(y_values) > 0:
            ax.plot(
                x_values[: len(y_values)],
                y_values,
                label=label,
                linestyle=linestyle,
                color=color,
            )

    # Generate distinct colors for models
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_dict)))

    # Training and Validation Loss Plots
    for (lr, data), color in zip(models_dict.items(), colors):
        safe_plot(
            axs[0, 0],
            epochs,
            data.get("train_loss", []),
            f"Train Loss (LR {lr})",
            "--",
            color=color,
        )
        safe_plot(
            axs[0, 1],
            epochs,
            data.get("val_loss", []),
            f"Val Loss (LR {lr})",
            "-",
            color=color,
        )

    axs[0, 0].set_title("Training Loss Across Learning Rates")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].set_title("Validation Loss Across Learning Rates")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Training and Validation Accuracy Plots
    for (lr, data), color in zip(models_dict.items(), colors):
        safe_plot(
            axs[1, 0],
            epochs,
            data.get("train_accuracy", []),
            f"Train Acc (LR {lr})",
            "--",
            color=color,
        )
        safe_plot(
            axs[1, 1],
            epochs,
            data.get("val_accuracy", []),
            f"Val Acc (LR {lr})",
            "-",
            color=color,
        )

    axs[1, 0].set_title("Training Accuracy Across Learning Rates")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Accuracy (%)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].set_title("Validation Accuracy Across Learning Rates")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("Accuracy (%)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Ensure integer epoch labels
    for ax_row in axs:
        for ax in ax_row:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def average_runs(data1, data2):
    """Compute the average between two runs for each key (handling missing data)."""
    averaged_data = {}

    for key in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
        list1 = data1.get(key, []) if data1 else []
        list2 = data2.get(key, []) if data2 else []

        # Find the max available length
        max_len = max(len(list1), len(list2))

        # Pad with NaNs to handle uneven lengths
        arr1 = np.array(list1 + [np.nan] * (max_len - len(list1)))
        arr2 = np.array(list2 + [np.nan] * (max_len - len(list2)))

        # Compute mean while ignoring NaNs
        averaged_data[key] = np.nanmean(np.vstack((arr1, arr2)), axis=0).tolist()

    return averaged_data


# Define the model
model = "densenet161"
learning_rate_models = {}
weight_decay_models = {}

# Iterate over learning rates and load both runs
learning_rates = ["1e-5"]
weight_decays = ["1e-2", "1e-4", "1e-6"]
"""for lr in learning_rates:
    run1 = None
    run2 = None
    #if lr not in ["1e-5", "5e-6"]:
    run1 = load_json_data(f"models/{model}/{model}_RGB_lr{lr}_run1_metrics.json")
    #if lr not in ["1e-4"]:
    run2 = load_json_data(f"models/{model}/{model}_RGB_lr{lr}_run12_metrics.json")

    # If both runs exist, average them; otherwise, take the existing one
    if run1 or run2:
        learning_rate_models[lr] = average_runs(run1, run2)"""

for wd in weight_decays:
    run1 = None
    run2 = None
    print(os.getcwd())
    run1 = load_json_data(
        f"models/temp_models/{model}_IRG_lr1e-5_wd{wd}_unfreezeNone_run1_metrics.json"
    )
    run2 = load_json_data(
        f"models/temp_models/{model}_IRG_lr1e-5_wd{wd}_unfreezeNone_run12_metrics.json"
    )

    # If both runs exist, average them; otherwise, take the existing one
    if run1 or run2:
        weight_decay_models[wd] = average_runs(run1, run2)


# Call the function to plot
def main():
    for learning_rate, data in weight_decay_models.items():
        if data:  # Ensure data is not empty
            # Find the index of the lowest validation loss
            lowest_val_loss_idx = data["val_loss"].index(min(data["val_loss"]))

            # Get values from the same epoch
            lowest_val_loss = data["val_loss"][lowest_val_loss_idx]
            corresponding_train_loss = data["train_loss"][lowest_val_loss_idx]
            corresponding_val_acc = data["val_accuracy"][lowest_val_loss_idx]
            corresponding_train_acc = data["train_accuracy"][lowest_val_loss_idx]

            print(
                f"Learning Rate: {learning_rate}, "
                f"Val Acc (At Min Val Loss Epoch): {corresponding_val_acc}, "
                f"Train Acc (At Min Val Loss Epoch): {corresponding_train_acc}, "
                f"Lowest Val Loss: {lowest_val_loss}, "
                f"Train Loss (At Min Val Loss Epoch): {corresponding_train_loss}"
            )

    # write_to_excel()
    """if learning_rate_models:
        plot_learning_rates(learning_rate_models)
    else:
        print("Could not load any JSON data. Please check file paths.")
    return"""


def write_to_excel():
    import pandas as pd

    file_path = "training_results_freeze_eb5.xlsx"

    results = []  # List to store results

    # Process weight decays
    for model in ["efficientnet_b5"]:
        for IMG_TYPE in ["RGB", "IRG", "RGB_NDVI"]:
            for lr in learning_rates:
                for wd in weight_decays:
                    run1 = load_json_data(
                        f"models/{model}_{IMG_TYPE}_lr{lr}_wd{wd}_unfreezeGradual_run1_metrics.json"
                    )
                    run2 = load_json_data(
                        f"models/{model}_{IMG_TYPE}_lr{lr}_wd{wd}_unfreezeGradual_run12_metrics.json"
                    )

                    # If run1 exists, store it
                    if run1:
                        lowest_val_loss_idx = run1["val_loss"].index(
                            min(run1["val_loss"])
                        )
                        results.append(
                            {
                                "Model": model,
                                "IMG": IMG_TYPE,
                                "LR": lr,
                                "WD": wd,
                                "Run": "Run 1",
                                "Freeze": True,
                                "Val.Acc.": run1["val_accuracy"][lowest_val_loss_idx],
                                "Train.Acc.": run1["train_accuracy"][
                                    lowest_val_loss_idx
                                ],
                                "Val.Loss": run1["val_loss"][lowest_val_loss_idx],
                                "Train.Loss": run1["train_loss"][lowest_val_loss_idx],
                                "Test.Acc": None,
                                "Test.F1": None,
                            }
                        )

                    # If run2 exists, store it
                    if run2:
                        lowest_val_loss_idx = run2["val_loss"].index(
                            min(run2["val_loss"])
                        )
                        results.append(
                            {
                                "Model": model,
                                "IMG": IMG_TYPE,
                                "LR": lr,
                                "WD": wd,
                                "Run": "Run 2",
                                "Freeze": True,
                                "Val.Acc.": run2["val_accuracy"][lowest_val_loss_idx],
                                "Train.Acc.": run2["train_accuracy"][
                                    lowest_val_loss_idx
                                ],
                                "Val.Loss": run2["val_loss"][lowest_val_loss_idx],
                                "Train.Loss": run2["train_loss"][lowest_val_loss_idx],
                                "Test.Acc": None,
                                "Test.F1": None,
                            }
                        )

    # Convert to DataFrame
    df_new = pd.DataFrame(results)

    # Save to Excel
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        df_combined = pd.concat([existing_df, df_new], ignore_index=True)
    else:
        df_combined = df_new  # No existing file, just use new data

    # Save back to Excel (preserving old data)
    df_combined.to_excel(file_path, index=False)

    print("Results saved to training_results_freeze.xlsx")

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Directory where result files are stored
RESULTS_DIR = "./results_3classes"  # Change this if needed
FINAL_RESULTS_FILE = "final_results.txt"

def load_results():
    """Loads all result files and returns a dictionary with data per model."""
    results_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.txt"))
    model_results = {}

    for file in results_files:
        model_name = os.path.splitext(os.path.basename(file))[0].replace("results_", "")
        df = pd.read_csv(file)

        if "label" in df.columns and "prediction" in df.columns:
            model_results[model_name] = df
        else:
            print(f"Skipping file {file} (invalid format)")

    return model_results


def majority_voting(model_results):
    """Performs majority voting across model predictions and computes confidence scores.
       If there is a tie, it defaults to Class 1 (Very High Risk)."""
    
    models = list(model_results.keys())
    first_model = models[0]
    
    # Use the first model's labels as ground truth (assuming all have the same labels)
    y_true = model_results[first_model]["label"].values
    num_samples = len(y_true)

    # Collect predictions from all models
    all_predictions = np.array([model_results[model]["prediction"].values for model in models])

    # Perform majority voting and compute confidence
    y_ensemble = []
    confidences = []
    
    for i in range(num_samples):
        votes = all_predictions[:, i]  # Get all model predictions for sample i
        vote_counts = Counter(votes)  # Count votes for each class
        
        # Get the two most common labels
        most_common = vote_counts.most_common(2)  # Get top 2 most frequent labels
        
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:  # If it's a tie
            most_common_label = 1  # Default to "Very High Risk"
            most_common_count = most_common[0][1]  # Since it's a tie, both have the same count
        else:
            most_common_label, most_common_count = most_common[0]  # Majority vote

        confidence = (most_common_count / len(models)) * 100  # Convert to percentage
        
        y_ensemble.append(most_common_label)
        confidences.append(confidence)

    return y_true, np.array(y_ensemble), np.array(confidences)


def compute_metrics(y_true, y_pred):
    """Computes accuracy and F1-score for the ensemble."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, f1

def plot_confusion_matrix(y_true, y_pred, normalize=False):
    """Plots and saves confusion matrix for the ensemble model with optional normalization."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize by row (actual class)
    
    plt.figure(figsize=(6,5))
    
    # Define custom labels
    class_labels = ["Low-Medium", "High", "Very High-Extreme"]
    
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Ensemble Model" + (" (Normalized)" if normalize else ""))
    
    # Save the plot
    plt.savefig("confusion_matrix_ensemble.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix_ensemble.png")


def save_final_results(y_true, y_pred, confidences, accuracy, f1_score):
    """Saves final ensemble results with accuracy, F1-score, and predictions including confidence."""
    with open(FINAL_RESULTS_FILE, "w") as f:
        f.write(f"Ensemble Model Accuracy: {accuracy:.4f}\n")
        f.write(f"Ensemble Model F1-Score: {f1_score:.4f}\n")
        f.write("\nlabel,prediction,confidence\n")
        for label, prediction, confidence in zip(y_true, y_pred, confidences):
            f.write(f"{label},{prediction},{confidence:.2f}%\n")

    print(f"Final results saved to {FINAL_RESULTS_FILE}")

def main():
    model_results = load_results()
    if not model_results:
        print("No valid result files found.")
        return

    y_true, y_ensemble, confidences = majority_voting(model_results)
    accuracy, f1 = compute_metrics(y_true, y_ensemble)

    plot_confusion_matrix(y_true, y_ensemble, True)
    save_final_results(y_true, y_ensemble, confidences, accuracy, f1)

    # Print final metrics
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    print(f"Ensemble F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()

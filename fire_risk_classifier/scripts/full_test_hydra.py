import csv
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from fire_risk_classifier.pipeline import Pipeline
from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.dataclasses.params import Params

irg_model_names = [] 
rgb_model_names = []
rgb_ndvi_model_names = [
    "resnet50_RGB_NDVI_lr1e-4_run1.pth",
    "resnet50_RGB_NDVI_lr1e-4_run2.pth",
    "resnet50_RGB_NDVI_lr1e-5_run2.pth",
    "resnet50_RGB_NDVI_lr1e-5_run1.pth",
    "resnet50_RGB_NDVI_lr5e-5_run1.pth",
    "resnet50_RGB_NDVI_lr5e-5_run2.pth",
    "resnet50_RGB_NDVI_lr5e-6_run1.pth",
    "resnet50_RGB_NDVI_lr5e-6_run2.pth",
]

NUM_CLASSES = 2
CALCULATE_IRG = True
CALCULATE_RGB = True
CALCULATE_RGB_NDVI = True


def majority_vote(all_predictions_list):
    """Takes predictions from all pipelines and performs majority voting."""
    combined_predictions = list(zip(*all_predictions_list))
    return [Counter(preds).most_common(1)[0][0] for preds in combined_predictions]


def calculate_confidence(all_predictions_list):
    """Calculates confidence as the median of prediction values per sample."""
    combined_predictions = np.array(all_predictions_list)
    return np.mean(combined_predictions, axis=0)


def write_final_predictions(filename, all_labels, final_predictions, confidences):
    """Writes final predictions along with labels and confidence scores to a CSV file."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Prediction", "Confidence"])  # Header
        for label, prediction, confidence in zip(
            all_labels, final_predictions, confidences
        ):
            writer.writerow([label, prediction, confidence])


def get_predictions(model: str, params: Params) -> list:
    """Runs the pipeline synchronously and gets predictions."""
    first_part = model.split("_")[0]
    if "d" in first_part:
        algo = "densenet161"
    elif "eff" in first_part:
        algo = "efficientnet_b5"
    elif "r" in first_part:
        algo = "resnet50"
    elif "test" in first_part:
        algo = "densenet161"
    args = {
        "test": True,
        "algorithm": algo,
        "load_weights": model,
    }
    pipeline = Pipeline(params=params, args=args)
    return pipeline.test_cnn(plot_confusion_matrix=False, log_info=False)


def get_params(is_irg: bool, is_ndvi: bool = False) -> Params:
    params = Params()
    base_images_path = "fire_risk_classifier/data/images/"
    params.directories["images_directory"] = (
        f"{base_images_path}ortos2018-IRG-62_5m-decompressed"
        if is_irg
        else f"{base_images_path}ortos2018-RGB-62_5m-decompressed"
    )
    params.calculate_ndvi_index = is_ndvi
    return params


def write_results(results: list, is_irg: bool):
    models = irg_model_names if is_irg else rgb_model_names
    for i, (labels, predictions) in enumerate(results):
        with open(f"results_{models[i]}_{'IRG' if is_irg else 'RGB'}.txt", "w") as f:
            f.write("label,prediction\n")
            for label, prediction in zip(labels, predictions):
                f.write(f"{label},{prediction}\n")


def main():
    all_labels_combined = None
    all_predictions_combined = []
    Logger.initialize_logger()

    if CALCULATE_IRG:
        # Run all pipelines and collect results
        params = get_params(is_irg=True)
        results = [get_predictions(model, params) for model in irg_model_names]
        #write_results(results, is_irg=True)

        """all_labels, all_predictions_list = zip(*results)
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)"""

    if CALCULATE_RGB:
        params = get_params(is_irg=False)
        results = [get_predictions(model, params) for model in rgb_model_names]
        #write_results(results, is_irg=False)

        """all_labels, all_predictions_list = zip(*results)
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)"""

    if CALCULATE_RGB_NDVI:
        params = get_params(is_irg=False, is_ndvi=True)
        results = [get_predictions(model, params) for model in rgb_ndvi_model_names]
        #write_results(results, is_irg=False)

        """all_labels, all_predictions_list = zip(*results)
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)"""

    #final_predictions = majority_vote(all_predictions_combined)
    #confidences = calculate_confidence(all_predictions_combined)

    #accuracy = sum(
    #    l == p for l, p in zip(all_labels_combined, final_predictions)
    #) / len(all_labels_combined)
    #print(f"Combined Accuracy: {accuracy:.2%}")

    # write_final_predictions("final_predictions.csv", all_labels_combined, final_predictions, confidences)
    # print("Final predictions written to final_predictions.csv")
    """import os
    import numpy as np
    import csv

    results_directory = "results"
    all_labels_combined = []
    predictions_per_sample = {}

    # Read predictions from all files
    for filename in os.listdir(results_directory):
        filepath = os.path.join(results_directory, filename)
        
        if filename.endswith(".txt"):  # Ensure it's a CSV file
            with open(filepath, "r") as f:
                lines = f.readlines()

                for idx, line in enumerate(lines[1:]):  # Skip header
                    label, prediction = line.strip().split(",")
                    label, prediction = int(label), int(prediction)  # Convert to int
                    
                    # Store labels (assumes all models use the same label order)
                    if idx >= len(all_labels_combined):
                        all_labels_combined.append(label)

                    # Store predictions per sample
                    if idx not in predictions_per_sample:
                        predictions_per_sample[idx] = []
                    
                    predictions_per_sample[idx].append(prediction)

    # Compute final predictions and confidences
    final_predictions = []
    confidences = []

    for idx in sorted(predictions_per_sample.keys()):
        preds = predictions_per_sample[idx]
        final_prediction = round(np.mean(preds))  # Majority voting (rounded mean)
        confidence = np.mean(preds)  # Confidence as average of predictions

        final_predictions.append(final_prediction)
        confidences.append(confidence)

    # Save to CSV
    output_file = "final_predictions2.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "final_prediction", "confidence"])  # Header
        for label, pred, conf in zip(all_labels_combined, final_predictions, confidences):
            writer.writerow([label, pred, conf])

    print(f"Final predictions written to {output_file}")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from sklearn.metrics import confusion_matrix

    def load_data(file_path):
        return pd.read_csv(file_path)

    def save_confusion_matrix(y_true, y_pred, class_names, file_name, folder_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {file_name}")

        # Save the image
        save_path = os.path.join(folder_path, f"{file_name}.png")
        plt.savefig(save_path)
        plt.close()

    folder_path = "results"  # Change this if needed
    class_names = ["LOW-HIGH", "VERY-HIGH-EXTREME"]

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            data = load_data(file_path)

            # Ensure column names are correct
            if "label" not in data.columns or "prediction" not in data.columns:
                raise ValueError(
                    f"{file} must contain 'label' and 'prediction' columns"
                )

            # Extract labels and predictions
            y_true = data["label"].values
            y_pred = data["prediction"].values

            # Save confusion matrix for each file
            save_confusion_matrix(
                y_true, y_pred, class_names, file.replace(".txt", ""), folder_path
            )
    """

if __name__ == "__main__":
    main()

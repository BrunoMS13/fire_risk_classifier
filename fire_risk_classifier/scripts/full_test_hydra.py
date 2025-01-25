from collections import Counter
from fire_risk_classifier.pipeline import Pipeline
from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.dataclasses.params import Params

model_names = [
    "densenet_body_2C.pth",
    "densenet169_body_2C.pth",
    "densenet201_body_2C.pth"
    # "densenet_NCW_FT_final_model.pth",
    # "resnet_CW_FT_final_model.pth",
    # "resnet_NCW_FT_final_model.pth",
]

CALCULATE_IRG = False
CALCULATE_RGB = True


def majority_vote(all_predictions_list):
    """Takes predictions from all pipelines and performs majority voting."""
    combined_predictions = list(zip(*all_predictions_list))
    return [Counter(preds).most_common(1)[0][0] for preds in combined_predictions]


def get_predictions(model: str, params: Params) -> list:
    """Runs the pipeline synchronously and gets predictions."""
    algo = model.split("_")[0]
    args = {
        "test": True,
        "algorithm": algo,
        "load_weights": model,
    }
    pipeline = Pipeline(params=params, args=args)
    return pipeline.test_cnn(plot_confusion_matrix=False)


def get_params(is_irg: bool) -> Params:
    params = Params()

    base_path = "fire_risk_classifier/data/cnn_checkpoint_weights/"
    params.directories["cnn_checkpoint_weights"] = (
        f"{base_path}IRG_2C/" if is_irg else f"{base_path}RGB_2C/"
    )

    base_images_path = "fire_risk_classifier/data/images/"
    params.directories["images_directory"] = (
        f"{base_images_path}ortos2018-IRG-62_5m-decompressed"
        if is_irg
        else f"{base_images_path}ortos2018-RGB-62_5m-decompressed"
    )

    return params


def write_results(results: list, is_irg: bool):
    for i, (labels, predictions) in enumerate(results):
        with open(
            f"results_{model_names[i]}_{'IRG' if is_irg else 'RGB'}.txt", "w"
        ) as f:
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
        results = [get_predictions(model, params) for model in model_names]
        write_results(results, is_irg=True)

        all_labels, all_predictions_list = zip(*results)
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)

    if CALCULATE_RGB:
        params = get_params(is_irg=False)
        results = [get_predictions(model, params) for model in model_names]
        write_results(results, is_irg=False)

        all_labels, all_predictions_list = zip(*results)
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)

    final_predictions = majority_vote(all_predictions_combined)

    accuracy = sum(
        l == p for l, p in zip(all_labels_combined, final_predictions)
    ) / len(all_labels_combined)
    print(f"Combined Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

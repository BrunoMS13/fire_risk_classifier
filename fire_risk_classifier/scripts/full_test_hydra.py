import asyncio
from collections import Counter
from fire_risk_classifier.pipeline import Pipeline

model_names = [
    "densenet_CW_FT_final_model.pth",
    "densenet_NCW_FT_final_model.pth",
    "resnet_CW_FT_final_model.pth",
    "resnet_NCW_FT_final_model.pth",
]

CALCULATE_IRG = True
CALCULATE_RGB = False


def majority_vote(all_predictions_list):
    """Takes predictions from all pipelines and performs majority voting."""
    combined_predictions = list(zip(*all_predictions_list))
    return [Counter(preds).most_common(1)[0][0] for preds in combined_predictions]


async def async_get_predictions(model_path: str) -> list:
    args = {
        "test": True,
        "load_weights": model_path,
        "algorithm": "densenet" if "densenet" in model_path else "resnet",
    }
    pipeline = Pipeline(args=args)
    return await pipeline.test_cnn(plot_confusion_matrix=False)


def main():
    classes = 2
    all_labels_combined = None
    all_predictions_combined = []

    if CALCULATE_IRG:
        irg_path = f"/models/IRG_{classes}C"
        paths = [f"{irg_path}/{model}" for model in model_names]

        # Run all pipelines and collect results
        predictions_coros = [async_get_predictions(path) for path in paths]
        results = asyncio.run(asyncio.gather(*predictions_coros))

        all_labels, all_predictions_list = zip(*results)

        # Assuming all pipelines use the same labels, we take the first set
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)

    if CALCULATE_RGB:
        rgb_path = f"/models/RGB_{classes}C"
        paths = [f"{rgb_path}/{model}" for model in model_names]

        # Run all pipelines and collect results
        predictions_coros = [async_get_predictions(path) for path in paths]
        results = asyncio.run(asyncio.gather(*predictions_coros))

        all_labels, all_predictions_list = zip(*results)

        # Assuming all pipelines use the same labels, we take the first set
        if all_labels_combined is None:
            all_labels_combined = all_labels[0]

        all_predictions_combined.extend(all_predictions_list)

    # Perform majority voting
    final_predictions = majority_vote(all_predictions_combined)

    # Print accuracy
    accuracy = sum(
        l == p for l, p in zip(all_labels_combined, final_predictions)
    ) / len(all_labels_combined)
    print(f"Combined Accuracy: {accuracy:.2%}")

import os
import csv
import numpy as np
from collections import Counter

from fire_risk_classifier.pipeline import Pipeline
from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.dataclasses.params import Params

irg_model_names = [
    "best_models/densenet161_IRG.pth",
    "best_models/efficientnetB5_IRG.pth",
    "best_models/resnet50_IRG.pth",
] 
rgb_model_names = [
    "best_models/densenet161_RGB.pth",
    "best_models/efficientnetB5_RGB.pth",
    "best_models/resnet50_RGB.pth",
]
rgb_ndvi_model_names = [
    "best_models/densenet161_RGB_NDVI.pth",
    "best_models/efficientnetB5_RGB_NDVI.pth",
    "best_models/resnet50_RGB_NDVI.pth",
]



NUM_CLASSES = 2
CALCULATE_IRG = True
CALCULATE_RGB = True
CALCULATE_RGB_NDVI = True


def get_predictions(model: str, params: Params) -> list:
    """Runs the pipeline synchronously and gets predictions."""
    algo = ""
    if "efficientnetB5" in model:
        algo = "efficientnet_b5"
    elif "densenet161" in model:
        algo = "densenet161"
    else:
        algo = "resnet50"

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


def write_results(results: list, image_type: str, model_paths: list):
    for model_path, (labels, predictions) in zip(model_paths, results):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        filename = f"results_{model_name}.txt"

        with open(filename, "w") as f:
            f.write("label,prediction\n")  # Header
            for label, prediction in zip(labels, predictions):
                f.write(f"{label},{prediction}\n")
        
        print(f"Results saved to {filename}")



def main():  # sourcery skip: extract-duplicate-method
    Logger.initialize_logger()

    if CALCULATE_IRG:
        params = get_params(is_irg=True)
        results = [get_predictions(model, params) for model in irg_model_names]
        write_results(results, "IRG", irg_model_names)

    if CALCULATE_RGB:
        params = get_params(is_irg=False)
        results = [get_predictions(model, params) for model in rgb_model_names]
        write_results(results, "RGB", rgb_model_names)

    if CALCULATE_RGB_NDVI:
        params = get_params(is_irg=False, is_ndvi=True)
        results = [get_predictions(model, params) for model in rgb_ndvi_model_names]
        write_results(results, "RGB_NDVI", rgb_ndvi_model_names)



if __name__ == "__main__":
    main()

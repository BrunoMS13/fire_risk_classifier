import os
import logging
import traceback
from PIL import Image
from torchvision import transforms

from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.utils.extractor import Extractor
from fire_risk_classifier.utils.kml_reader import KMLReader
from fire_risk_classifier.dataclasses.house_point import HousePoint


augmentation_pipeline = transforms.Compose(
    [
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)


def augment_and_save_images(
    image_directory: str,
    output_directory: str,
    all_points: list[HousePoint],
    augment_classes: list[int],
    num_augments: int = 5,
):
    """
    Augments and saves images to a new directory.

    :param image_directory: Directory containing original images.
    :param output_directory: Directory to save augmented images.
    :param all_points: List of dicts with `image_id` and `fire_risk`.
    :param augment_classes: Classes to augment (default: None for all).
    :param num_augments: Number of augmented versions to create for each image.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    total_augments_per_class = {cls: 0 for cls in augment_classes}

    for point in all_points:
        image_id = point["image_id"]
        fire_risk = point["fire_risk"]

        if fire_risk not in augment_classes:
            continue

        total_augments_per_class[fire_risk] += num_augments

        # Load image
        image_path = os.path.join(image_directory, f"{image_id}.png")
        if not os.path.exists(image_path):
            logging.warning(f"Image {image_path} not found. Skipping.")
            continue

        image = Image.open(image_path)

        # Save the original image
        # image.save(os.path.join(output_directory, f"{image_id}.jpg"))

        # Only augment specified classes (if provided)
        for i in range(num_augments):
            # Apply augmentations
            augmented_image = augmentation_pipeline(image)

            # Save augmented image with a new name
            augmented_image_path = os.path.join(
                output_directory, f"{image_id}_aug{i + 1}.png"
            )
            augmented_image.save(augmented_image_path)
            logging.info(f"Saved augmented image: {augmented_image_path}")

    logging.info(f"Total augmented images per class: {total_augments_per_class}")


def main():
    # Configure logging to display INFO level messages and above
    Logger.initialize_logger()
    try:
        # logging.info("Starting the application...")

        image_directory = "images/ortos2018-IRG-decompressed"
        kmz_path = "fire_risk_classifier/data/Pontos.kmz"

        extractor = Extractor(kmz_path)
        kml_file = extractor.get_file_data()
        kml_reader = KMLReader(kml_file)
        data = kml_reader.get_kml_data()

        all_points = []
        for _, points in data.items():
            all_points.extend(
                {"image_id": point.name, "fire_risk": point.fire_risk.value}
                for point in points
            )
        augment_and_save_images(
            image_directory, "images/augmented-2018-IRG/low", all_points, [0]
        )

        # augment_classes = [0, 1]

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

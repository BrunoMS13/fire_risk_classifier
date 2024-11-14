import logging
import traceback
from fire_risk_classifier.data.image_dataset import CustomImageDataset
from fire_risk_classifier.utils.logger import Logger


def main():
    # Configure logging to display INFO level messages and above
    Logger.initialize_logger()
    try:
        logging.info("Starting the application...")

        image_directory = "images/ortos2018-IRG-decompressed"
        annotations_file = "fire_risk_classifier/data/csvs/train.csv"

        dataset = CustomImageDataset(
            annotations_file,
            image_directory,
        )
        # dataset.__getitem__(0)
        for i in range(20):
            dataset.__getitem__(i)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

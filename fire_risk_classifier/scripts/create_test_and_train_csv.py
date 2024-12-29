import os
import csv
import random
from typing import Any

from fire_risk_classifier.utils.extractor import Extractor
from fire_risk_classifier.utils.kml_reader import KMLReader
from fire_risk_classifier.dataclasses.house_point import HousePoint


RANDOM_SEED = 42
TRAIN_RATIO = 0.7

medium_risks = [0, 1, 2]
extreme_risks = [3, 4]


def get_all_points() -> list[dict[str, Any]]:
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
    return all_points


def write_csv(data: list[dict[str, Any]], output_file: str):
    csv_path = "fire_risk_classifier/data/csvs"
    os.makedirs(csv_path, exist_ok=True)
    output_path = os.path.join(csv_path, output_file)

    updated_data = []
    for d in data:
        risk = d["fire_risk"]

        if risk in medium_risks:
            # print(f"Risk of {risk} is medium")
            updated_data.append({"image_id": d["image_id"], "fire_risk": 0})
            continue
        # print(f"Risk of {risk} is extreme")
        risk = 1
        updated_data.append({"image_id": d["image_id"], "fire_risk": 1})

    data = updated_data

    with open(output_path, mode="w", newline="") as csv_file:
        fieldnames = ["image_id", "fire_risk"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    random.seed(RANDOM_SEED)

    all_points = get_all_points()
    random.shuffle(all_points)

    train_size = int(len(all_points) * TRAIN_RATIO)
    train_data = all_points[:train_size]
    test_data = all_points[train_size:]

    write_csv(train_data, "train_2classes.csv")
    write_csv(test_data, "test_2classes.csv")

    print("CSV files created successfully.")

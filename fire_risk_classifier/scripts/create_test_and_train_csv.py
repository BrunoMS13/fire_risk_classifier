import os
import csv
import random
from typing import Any
from collections import defaultdict

from fire_risk_classifier.utils.extractor import Extractor
from fire_risk_classifier.utils.kml_reader import KMLReader
from fire_risk_classifier.dataclasses.house_point import HousePoint


RANDOM_SEED = 42
TRAIN_RATIO = 0.7

low_risk = [0, 1]
medium_risks = [2]
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


def get_parsed_data(data: list[dict[str, Any]], n_classes: int) -> tuple:
    updated_data, data_distribution = [], defaultdict(int)
    for d in data:
        risk = d["fire_risk"]

        if n_classes == 2:
            if risk in low_risk + medium_risks:
                updated_data.append({"image_id": d["image_id"], "fire_risk": 0})
                data_distribution[0] += 1
                continue
            data_distribution[1] += 1
            updated_data.append({"image_id": d["image_id"], "fire_risk": 1})
        elif n_classes == 3:
            if risk in low_risk:
                updated_data.append({"image_id": d["image_id"], "fire_risk": 0})
                data_distribution[0] += 1
                continue
            if risk in medium_risks:
                updated_data.append({"image_id": d["image_id"], "fire_risk": 1})
                data_distribution[1] += 1
                continue
            data_distribution[2] += 1
            updated_data.append({"image_id": d["image_id"], "fire_risk": 2})
        else:
            raise ValueError("Invalid number of classes. Please select 2 or 3.")
    return updated_data, data_distribution


def write_csv(data: list[dict[str, Any]], output_file: str, n_classes: int) -> None:
    csv_path = "fire_risk_classifier/data/csvs"
    os.makedirs(csv_path, exist_ok=True)
    output_path = os.path.join(csv_path, output_file)

    data, data_distribution = get_parsed_data(data, n_classes)
    print(data_distribution)

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
    n_classes = input("How many classes do you want to create? [Options: 2 or 3]\n")

    write_csv(train_data, f"train_{n_classes}classes.csv", int(n_classes))
    write_csv(test_data, f"test_{n_classes}classes.csv", int(n_classes))

    print("CSV files created successfully.")

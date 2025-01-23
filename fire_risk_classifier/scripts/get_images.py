import os
import logging
import asyncio
import aiofiles
import traceback

from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.utils.extractor import Extractor
from fire_risk_classifier.utils.kml_reader import KMLReader
from fire_risk_classifier.utils.image_getter import WMSFetcher
from fire_risk_classifier.dataclasses.enums import RegionsEnum
from fire_risk_classifier.dataclasses.house_point import HousePoint
from fire_risk_classifier.dataclasses.bounding_box import BoundingBox


BATCH_SIZE = 20
BBOX_SIZE = 50
IMAGE_TYPE = "IRG"  # Can be "IRG" or "RGB"


async def save_map_image(
    bbox: BoundingBox,
    fetcher: WMSFetcher,
    house_point: HousePoint,
):
    directory_path = f"fire_risk_classifier/data/images/ortos2018-{IMAGE_TYPE}-50m"
    os.makedirs(directory_path, exist_ok=True)

    image_data = await fetcher.get_map(bbox=bbox)
    file_path = f"{directory_path}/{house_point.name}.png"

    async with aiofiles.open(file_path, mode="wb") as file:
        await file.write(image_data)
    print(f"Saved {file_path}")


def get_missing_regions_data(data: dict[str, list[HousePoint]]):
    points = []
    for region in RegionsEnum.__members__.values():
        points.extend(data[region.value])
    return points


async def save_map_images(
    house_points: list[HousePoint], fetcher: WMSFetcher, bboxes: list[BoundingBox]
):
    total = len(house_points)
    for start in range(0, total, BATCH_SIZE):
        await asyncio.gather(
            *[
                save_map_image(bbox, fetcher, house_point)
                for house_point, bbox in zip(
                    house_points[start : start + BATCH_SIZE],
                    bboxes[start : start + BATCH_SIZE],
                )
            ]
        )


def get_house_point_data() -> dict[str, list[HousePoint]]:
    kmz_path = "fire_risk_classifier/data/Pontos.kmz"

    extractor = Extractor(kmz_path)
    kml_file = extractor.get_file_data()

    if not kml_file:
        return

    kml_reader = KMLReader(kml_file)
    return kml_reader.get_kml_data()


# In case you need to rename the images.
def rename_images(directory: str):
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file contains the substring to be removed
        if "-RGB" in filename:
            # Create the new filename by replacing the substring
            new_filename = filename.replace("-RGB", "")
            # Construct full paths for old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} -> {new_file}")


def main():
    # Configure logging to display INFO level messages and above
    Logger.initialize_logger()
    logging.info("Starting the application...")

    data = get_house_point_data()

    base_url = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018"
    fetcher = WMSFetcher(base_url)

    house_points = get_missing_regions_data(data)
    bboxes = [house_point.get_bbox(BBOX_SIZE) for house_point in house_points]

    asyncio.run(save_map_images(house_points, fetcher, bboxes))
    logging.info("Map images saved.")


# https://cartografia.dgterritorio.gov.pt/wms/ortos2018?language=por&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX=38.64901179894537%2C-9.224836368150017%2C38.64991112055128%2C-9.223684840896238&CRS=EPSG%3A4326&WIDTH=812&HEIGHT=800&LAYERS=Ortos&STYLES=&FORMAT=image%2Fpng&DPI=96&MAP_RESOLUTION=96&FORMAT_OPTIONS=dpi%3A96&TRANSPARENT=TRUE

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


async def save_map_image(
    bbox: BoundingBox,
    fetcher: WMSFetcher,
    house_point: HousePoint,
):
    directory_path = "fire_risk_classifier/data/images/ortos2018-IRG"
    os.makedirs(directory_path, exist_ok=True)

    image_data = await fetcher.get_map(bbox=bbox)
    file_path = f"{directory_path}/{house_point.name}.png"

    async with aiofiles.open(file_path, mode="wb") as file:
        await file.write(image_data)
    print(f"Saved {file_path}")


def main():
    # Configure logging to display INFO level messages and above
    Logger.initialize_logger()
    try:
        logging.info("Starting the application...")
        kmz_path = "fire_risk_classifier/data/Pontos.kmz"

        extractor = Extractor(kmz_path)
        kml_file = extractor.get_file_data()

        if not kml_file:
            return

        kml_reader = KMLReader(kml_file)
        data = kml_reader.get_kml_data()

        base_url = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018"
        fetcher = WMSFetcher(base_url)

        def get_missing_regions_data():
            points = []
            for region in RegionsEnum.__members__.values():
                points.extend(data[region.value])
            return points

        house_points = get_missing_regions_data()
        bboxes = [house_point.get_bbox(50) for house_point in house_points]

        async def async_call():
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

        asyncio.run(async_call())

        logging.info("Map images saved.")

        # https://cartografia.dgterritorio.gov.pt/wms/ortos2018?language=por&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX=38.64901179894537%2C-9.224836368150017%2C38.64991112055128%2C-9.223684840896238&CRS=EPSG%3A4326&WIDTH=812&HEIGHT=800&LAYERS=Ortos&STYLES=&FORMAT=image%2Fpng&DPI=96&MAP_RESOLUTION=96&FORMAT_OPTIONS=dpi%3A96&TRANSPARENT=TRUE

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

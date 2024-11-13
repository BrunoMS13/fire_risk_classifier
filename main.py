import logging
import zipfile
import traceback
from typing import Iterable
from dataclasses import dataclass
from math import cos, pi, radians

from fire_risk_classifier.dataclasses.bounding_box import BoundingBox
from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.utils.image_getter import WMSFetcher
from fire_risk_classifier.utils.extractor import Extractor
from fire_risk_classifier.utils.kml_reader import KMLReader
from fire_risk_classifier.dataclasses.house_point import HousePoint


def main():
    # Configure logging to display INFO level messages and above
    Logger.initialize_logger()
    try:
        logging.info("Starting the application...")
        kmz_path = "fire_risk_classifier/data/Pontos.kmz"

        extractor = Extractor(kmz_path)
        kml_file = extractor.get_file_data()

        if not kml_file:
            print("No KML file found in the KMZ archive.")
            return

        kml_reader = KMLReader(kml_file)
        data = kml_reader.get_kml_data()

        # base_url = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018"
        # fetcher = WMSFetcher(base_url)

        # point = "SE4"
        print(data)

        """bbox = data[point].get_bbox(50)
        print(bbox)

        image_data = fetcher.get_map(bbox=bbox)
        # Save the image to a file
        with open("fire_risk_classifier/data/images/map_image.png", "wb") as image_file:
            image_file.write(image_data)
        logging.info("Map image saved.")"""

        # https://cartografia.dgterritorio.gov.pt/wms/ortos2018?language=por&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX=38.64901179894537%2C-9.224836368150017%2C38.64991112055128%2C-9.223684840896238&CRS=EPSG%3A4326&WIDTH=812&HEIGHT=800&LAYERS=Ortos&STYLES=&FORMAT=image%2Fpng&DPI=96&MAP_RESOLUTION=96&FORMAT_OPTIONS=dpi%3A96&TRANSPARENT=TRUE

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

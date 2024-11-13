import shutil
import logging
import xml.etree.ElementTree as ET

from fire_risk_classifier.dataclasses.enums import RegionsEnum
from fire_risk_classifier.dataclasses.house_point import HousePoint


class KMLReader:
    def __init__(self, kml_file: str | None):
        self.kml_file = kml_file

    def get_kml_data(self) -> dict[str, HousePoint]:
        logging.info(f"Reading KML file: {self.kml_file}")
        if not self.kml_file:
            raise Exception("KML file is None.")
        tree = ET.parse(self.kml_file)
        root = tree.getroot()

        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        data = self.__create_data_dict()

        for placemark in root.findall(".//kml:Placemark", ns):
            name = self.__get_text_from_placemark(placemark, "kml:name", ns)
            point = self.__get_text_from_placemark(
                placemark, ".//kml:Point/kml:coordinates", ns
            )
            style_url = self.__get_text_from_placemark(placemark, "kml:styleUrl", ns)

            if name is not None and point is not None:
                coordinates = point.split(",")
                house_point = HousePoint(name, *coordinates, style_url)
                data[self.__remove_integers_from_string(name)].append(house_point)
        return data

    def __create_data_dict(self) -> dict[RegionsEnum, list[HousePoint]]:
        data = {}
        for _, value in RegionsEnum.__members__.items():
            data[value.value] = []
        return data

    def __remove_integers_from_string(self, string: str) -> str:
        return "".join([char for char in string if not char.isdigit()])

    def __get_text_from_placemark(
        self, placemark: ET.Element, tag: str, ns: dict[str, str]
    ) -> str:
        element = placemark.find(tag, ns)
        return element.text if element is not None else ""

    def clean_up(self):
        """Cleans up the temporary directory."""
        shutil.rmtree(self.temp_dir)

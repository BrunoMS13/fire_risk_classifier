import os
import zipfile
import logging


class Extractor:

    def __init__(self, path: str):
        self.path = path
        self.temp_dir = self.__create_temp_dir()

    def get_file_data(self) -> str | None:
        logging.info(f"Extracting KMZ file: {self.path}")
        self.__extract_kmz_file()
        return self.__find_kml_file(self.temp_dir)

    def __extract_kmz_file(self):
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(self.temp_dir)

    def __create_temp_dir(self) -> str:
        temp_dir = f"data/kml_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        return temp_dir

    def __find_kml_file(self, temp_dir: str) -> str | None:
        kml_file = None
        for filename in os.listdir(temp_dir):
            if filename.endswith(".kml"):
                kml_file = os.path.join(temp_dir, filename)
                break
        return kml_file

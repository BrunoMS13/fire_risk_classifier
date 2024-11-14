import math
from dataclasses import dataclass

from fire_risk_classifier.dataclasses.enums import FireRisk
from fire_risk_classifier.dataclasses.bounding_box import BoundingBox


@dataclass
class HousePoint:
    name: str
    longitude: float
    latitude: float
    altitude: float
    fire_risk: FireRisk

    def __init__(
        self,
        name: str,
        longitude: str,
        latitude: str,
        altitude: str,
        style_url: str,
    ):
        self.name = name
        self.latitude = float(latitude)
        self.altitude = float(altitude)
        self.longitude = float(longitude)
        self.fire_risk = FireRisk.from_style_url(style_url, name)

    def get_bbox(self, radius: int) -> BoundingBox:
        # Earth's radius in meters
        earth_radius = 6371000

        # Convert radius to degrees
        delta_lat = (radius / earth_radius) * (180 / math.pi)
        delta_lon = (
            (radius / earth_radius)
            * (180 / math.pi)
            / math.cos(math.radians(self.latitude))
        )

        min_lat = self.latitude - delta_lat
        max_lat = self.latitude + delta_lat
        min_lon = self.longitude - delta_lon
        max_lon = self.longitude + delta_lon

        return BoundingBox(min_lon, min_lat, max_lon, max_lat)

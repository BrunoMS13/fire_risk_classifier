import math

from typing import Iterable
from dataclasses import dataclass


@dataclass
class BoundingBox:
    min_longitude: float
    min_latitude: float
    max_longitude: float
    max_latitude: float

    def __iter__(self) -> Iterable[float]:
        return iter(
            (
                self.min_latitude,
                self.min_longitude,
                self.max_latitude,
                self.max_longitude,
            )
        )

    def __str__(self):
        return f"{self.min_latitude},{self.min_longitude},{self.max_latitude},{self.max_longitude}"

    def get_distance(lat1: int, lon1: int, lat2: int, lon2: int) -> float:
        """Distance between two points on Earth's surface using the Haversine formula."""
        R = 6371  # Earth radius in kilometers
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

import re
from enum import Enum


class FireRisk(Enum):
    LOW = 0  # Blue
    MEDIUM = 1  # Green
    HIGH = 2  # Yellow
    VERY_HIGH = 3  # Orange (White pointer for some reason).
    EXTREME = 4  # Red
    UNKNOWN = 5

    @staticmethod
    def from_style_url(style_url: str, name) -> "FireRisk":
        if match := re.search(r"#m(?:sn)?_([a-z]+)-", style_url, re.IGNORECASE):
            color_code = match[1].upper()
            # Map the extracted part to FireRisk levels.
            mapping = {
                "BLU": FireRisk.LOW,
                "GRN": FireRisk.MEDIUM,
                "YLW": FireRisk.HIGH,
                "WHT": FireRisk.VERY_HIGH,
                "RED": FireRisk.EXTREME,
                "PURPLE": FireRisk.LOW,  # Weird edge case C61
                "LTBLU": FireRisk.VERY_HIGH,  # Weird edge case A82
            }
            return mapping.get(color_code, FireRisk.UNKNOWN)
        return FireRisk.UNKNOWN


class RegionsEnum(Enum):
    VIANA_DO_CASTELO = "VC"
    BRAGA = "B"
    VILA_REAL = "VR"
    BRAGANCA = "BR"
    PORTO = "P"
    AVEIRO = "A"
    VISEU = "V"
    GUARDA = "G"
    COIMBRA = "C"
    CASTELO_BRANCO = "CB"
    LEIRIA = "LE"
    SANTAREM = "SA"
    PORTALEGRE = "PO"
    LISBOA = "L"
    SETUBAL = "SE"
    EVORA = "E"
    BEJA = "BE"
    FARO = "F"

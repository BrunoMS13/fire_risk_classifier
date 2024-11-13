import aiohttp


class WMSFetcher:
    def __init__(self, base_url):
        self.base_url = base_url

    async def get_map(
        self,
        bbox,
        crs: str = "EPSG:4326",
        width=800,
        height=800,
        layers="Ortos2018-IRG",
        format="image/png",
        transparent="TRUE",
        version="1.3.0",
    ):
        """Fetches a map image from the WMS service.
        Args:
            bbox (tuple): Bounding box (minx, miny, maxx, maxy).
            crs (str, optional): Coordinate Reference System. Defaults to 'EPSG:4326'.
            width (int, optional): Width of the image. Defaults to 800.
            height (int, optional): Height of the image. Defaults to 800.
            layers (str, optional): Layers to display. Defaults to 'Ortos2018-IRG'.
            format (str, optional): Image format. Defaults to 'image/png'.
            transparent (str, optional): Whether the image background should be transparent. Defaults to 'TRUE'.
            version (str, optional): WMS version. Defaults to '1.3.0'.
        Returns:
            bytes: The image data.
        """
        params = {
            "SERVICE": "WMS",
            "VERSION": version,
            "REQUEST": "GetMap",
            "BBOX": ",".join(map(str, bbox)),
            "CRS": crs,
            "WIDTH": width,
            "HEIGHT": height,
            "LAYERS": layers,
            "FORMAT": format,
            "DPI": "96",  # Consistent with the method default, adjust if necessary
            "MAP_RESOLUTION": "96",  # Consistent with the method default, adjust if necessary
            "FORMAT_OPTIONS": "dpi:96",
            "TRANSPARENT": transparent,
        }
        # base_url = "https://cartografia.dgterritorio.gov.pt/wms/ortos2018"
        # generated_url = requests.Request("GET", base_url, params=params).prepare().url

        # https://cartografia.dgterritorio.gov.pt/wms/ortos2018?language=por&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX=38.64901179894537%2C-9.224836368150017%2C38.64991112055128%2C-9.223684840896238&CRS=EPSG%3A4326&WIDTH=812&HEIGHT=800&LAYERS=Ortos&STYLES=&FORMAT=image%2Fpng&DPI=96&MAP_RESOLUTION=96&FORMAT_OPTIONS=dpi%3A96&TRANSPARENT=TRUE
        # https://cartografia.dgterritorio.gov.pt/wms/ortos2018?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&BBOX=38.64901179894537%2C-9.224836368150017%2C38.64991112055128%2C-9.223684840896238&CRS=EPSG%3A4326&WIDTH=800&HEIGHT=800&LAYERS=Ortos2018-IRG&STYLES=&FORMAT=image%2Fpng&DPI=96&MAP_RESOLUTION=96&FORMAT_OPTIONS=dpi%3A96&TRANSPARENT=TRUE
        # response = requests.get(self.base_url, params=params)
        # response.raise_for_status()  # Raises an HTTPError if the response was an error

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                response.raise_for_status()
                return await response.read()

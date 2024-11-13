import logging


class Logger:

    @staticmethod
    def initialize_logger():
        logging.basicConfig(
            level=logging.INFO,
            format="\n[%(levelname)s] %(asctime)s\n%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

import logging
import argparse
from typing import Any

from fire_risk_classifier.pipeline import Pipeline
from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.dataclasses.params import Params


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Fire Risk Classifier argument parser for training and testing"
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def add_arguments(self):
        self.parser.add_argument("--algorithm", default="", type=str)
        self.parser.add_argument("--train", default="", type=str)
        self.parser.add_argument("--test", default="", type=str)
        self.parser.add_argument("--nm", default="", type=str)
        self.parser.add_argument("--prepare", default="", type=str)
        self.parser.add_argument("--num_gpus", default=1, type=int)
        self.parser.add_argument("--num_epochs", default=12, type=int)
        self.parser.add_argument("--batch_size", default=8, type=int)
        self.parser.add_argument("--load_weights", default="", type=str)
        self.parser.add_argument("--fine_tunning", default="", type=str)
        self.parser.add_argument("--class_weights", default="", type=str)
        self.parser.add_argument("--prefix", default="", type=str)
        self.parser.add_argument("--generator", default="", type=str)
        self.parser.add_argument("--database", default="", type=str)
        self.parser.add_argument("--path", default="", type=str)

    def get_parser_dict(self) -> dict[str, Any]:
        return vars(self.parser.parse_args())

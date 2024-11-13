import os
import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from fire_risk_classifier.dataclasses.params import Params
from fire_risk_classifier.classifier.cnn import get_cnn_model
from fire_risk_classifier.data.image_dataset import CustomImageDataset


class Pipeline:
    def __init__(self, params: Params = None, args: dict = {}):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return:
        """
        self.params = params

        if args["train"]:
            self.params.train_cnn = True
        if args["path"]:
            self.params.path = args["path"]
        """if parser["prepare"]:
            prepare_data(params)"""
        if args["algorithm"]:
            self.params.algorithm = args["algorithm"]
        if args["nm"]:
            self.params.use_metadata = False
        if args["test"]:
            self.params.test_cnn = True
        if args["num_gpus"]:
            self.params.num_gpus = args["num_gpus"]
        if args["load_weights"]:
            self.params.model_weights = args["load_weights"]
        if args["num_epochs"]:
            self.params.cnn_epochs = args["num_epochs"]
        if args["batch_size"]:
            self.params.batch_size_cnn = args["batch_size"]
        if args["fine_tunning"]:
            self.params.fine_tunning = True
        if args["class_weights"]:
            self.params.class_weights = args["class_weights"]

        image_directory = "images/ortos2018-IRG-decompressed"
        annotations_file = "output.csv"

        self.dataset = CustomImageDataset(
            annotations_file,
            image_directory,
        )
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.params.batch_size_cnn, shuffle=True
        )
        print(f"Batch Size: {self.params.batch_size_cnn}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_cnn(self):
        model = get_cnn_model(self.params).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0

        def lr_lambda(step: int):
            if self.params.lr_mode == "progressive_drops":
                if self.current_epoch == int(0.75 * self.params.cnn_epochs):
                    scale_factor = 0.01
                elif self.current_epoch == int(0.15 * self.params.cnn_epochs):
                    scale_factor = 0.1
                else:
                    scale_factor = 1
            return scale_factor

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        if self.params.model_weights:
            model.load_state_dict(torch.load(self.params.model_weights))

        for epoch in range(self.params.cnn_epochs):
            self.current_epoch = epoch
            self.training_step(epoch, model, optimizer, scheduler, criterion)
            torch.cuda.empty_cache()

        final_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"], "final_model.pth"
        )
        torch.save(model.state_dict(), final_path)

    def training_step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LambdaLR,
        criterion: nn.Module,
    ):
        model.train()

        total_samples = 0
        running_loss = 0.0
        correct_predictions = 0
        total_steps = len(self.data_loader)

        for step, (images, labels) in enumerate(self.data_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            print(
                f"Epoch [{epoch + 1}/{self.params.cnn_epochs}], "
                f"Step [{step + 1}/{total_steps}], "
                f"Loss: {running_loss / (step + 1):.4f}, "
                f"Accuracy: {100 * correct_predictions / total_samples:.2f}%, "
                f"LR: {scheduler.get_last_lr()[0]}"
            )
            # Save checkpoint
            if epoch % 5 == 0:  # Save every 5 epochs, adjust as needed
                self.__save_checkpoint(model, epoch)

        running_loss += loss.item()

    def __save_checkpoint(self, model: nn.Module, epoch: int):
        os.makedirs(self.params.directories["cnn_checkpoint_weights"], exist_ok=True)
        checkpoint_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"checkpoint_epoch_{epoch}.pth",
        )
        torch.save(model.state_dict(), checkpoint_path)


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Fire Risk Classifier argument parser for training and testing"
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def get_parser_dict(self) -> dict[str, Any]:
        return vars(self.parser.parse_args())


if __name__ == "__main__":
    default_params = Params()
    parser = ArgumentParser()

    parser.add_argument("--algorithm", default="", type=str)
    parser.add_argument("--train", default="", type=str)
    parser.add_argument("--test", default="", type=str)
    parser.add_argument("--nm", default="", type=str)
    parser.add_argument("--prepare", default="", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--num_epochs", default=12, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--load_weights", default="", type=str)
    parser.add_argument("--fine_tunning", default="", type=str)
    parser.add_argument("--class_weights", default="", type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--generator", default="", type=str)
    parser.add_argument("--database", default="", type=str)
    parser.add_argument("--path", default="", type=str)

    pipeline = Pipeline(default_params, parser.get_parser_dict())
    params = pipeline.params

    if params.train_cnn:
        pipeline.train_cnn()
    # if params.test_cnn:
    #    pipeline.test_cnn()
    ...

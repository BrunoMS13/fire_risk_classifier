import os
import logging
import argparse
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from fire_risk_classifier.dataclasses.params import Params
from fire_risk_classifier.classifier.cnn import get_cnn_model
from fire_risk_classifier.data.image_dataset import CustomImageDataset


class Pipeline:
    def __init__(self, params: Params = None, args: dict | None = None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return:
        """
        self.params = params

        if args is None:
            args = {}
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

        self.__init_data_loaders()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    def __init_data_loaders(self):
        directories = self.params.directories
        batch_size = self.params.batch_size_cnn

        if self.params.train_cnn:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            # Training data loader
            dataset = CustomImageDataset(
                directories["annotations_file"],
                directories["images_directory"],
                transform=transform,
            )
            self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return

        # Testing data loader
        test_dataset = CustomImageDataset(
            directories["testing_annotations_file"],
            directories["images_directory"],
        )
        self.test_data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

    def train_cnn(self):
        model = get_cnn_model(self.params).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0

        def lr_lambda(step: int):
            if self.params.lr_mode == "progressive_drops":
                if self.current_epoch >= int(0.75 * self.params.cnn_epochs):
                    scale_factor = 0.01
                elif self.current_epoch >= int(0.45 * self.params.cnn_epochs):
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

        logging.info(
            f"Saving final model to {self.params.directories['cnn_checkpoint_weights']}"
        )
        final_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.params.algorithm}_final_model.pth",
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

            running_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            logging.info(
                f"Epoch [{epoch + 1}/{self.params.cnn_epochs}], "
                f"Step [{step + 1}/{total_steps}], "
                f"Loss: {running_loss / (step + 1):.4f}, "
                f"Accuracy: {100 * correct_predictions / total_samples:.2f}%, "
                f"LR: {scheduler.get_last_lr()[0]}"
            )
            # Save checkpoint
            if epoch % 5 == 0 and step == 0:
                self.__save_checkpoint(model, epoch)

    def test_cnn(self):
        model = get_cnn_model(self.params).to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # Load saved model weights
        if self.params.model_weights:
            path = os.path.join(
                self.params.directories["cnn_checkpoint_weights"],
                self.params.model_weights,
            )
            model.load_state_dict(torch.load(path))
            logging.info(f"Loaded model weights from {self.params.model_weights}")

        total_samples = 0
        total_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():  # Disable gradient computation for testing
            for step, (images, labels) in enumerate(
                self.test_data_loader
            ):  # Use test data loader
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                if step % 10 == 0:
                    logging.info(
                        f"Test Step [{step + 1}/{len(self.test_data_loader)}], "
                        f"Loss: {total_loss / (step + 1):.4f}, "
                        f"Accuracy: {100 * correct_predictions / total_samples:.2f}%"
                    )

        final_loss = total_loss / len(self.test_data_loader)
        final_accuracy = 100 * correct_predictions / total_samples

        logging.info(f"Final Test Loss: {final_loss:.4f}")
        logging.info(f"Final Test Accuracy: {final_accuracy:.2f}%")

    def __save_checkpoint(self, model: nn.Module, epoch: int):
        os.makedirs(self.params.directories["cnn_checkpoint_weights"], exist_ok=True)
        logging.info(f"Saving checkpoint for epoch {epoch}")
        checkpoint_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.params.algorithm}_checkpoint_epoch_{epoch}.pth",
        )
        torch.save(model.state_dict(), checkpoint_path)

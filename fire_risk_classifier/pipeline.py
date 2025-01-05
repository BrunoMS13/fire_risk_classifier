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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            # Training data loader
            self.dataset = CustomImageDataset(
                directories["annotations_file"],
                directories["images_directory"],
                transform=transform,
            )
            self.data_loader = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True
            )

        if self.params.test_cnn:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Testing data loader
            self.test_dataset = CustomImageDataset(
                directories["testing_annotations_file"],
                directories["images_directory"],
                transform=transform,
            )
            self.test_data_loader = DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=True
            )

    def train_cnn(self):
        model = get_cnn_model(self.params).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.current_epoch = 0

        criterion = self.__get_criterion()
        scheduler = self.__get_scheduler(optimizer)

        if self.params.model_weights:
            model.load_state_dict(torch.load(self.params.model_weights))

        for epoch in range(self.params.cnn_epochs):
            self.current_epoch = epoch
            self.__training_step(epoch, model, optimizer, scheduler, criterion)
            torch.cuda.empty_cache()

        logging.info(
            f"Saving final model to {self.params.directories['cnn_checkpoint_weights']}"
        )
        final_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            self.__create_model_name(),
        )
        torch.save(model.state_dict(), final_path)

    def __create_model_name(self) -> str:
        # return f"{self.params.algorithm}_body_2C.pth"
        fine_tunned = "FT" if self.params.fine_tunning else "NFT"
        class_weights = "CW" if self.params.class_weights else "NCW"
        return f"{self.params.algorithm}_{class_weights}_{fine_tunned}_final_model.pth"

    def __training_step(
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
            # if epoch % 5 == 0 and step == 0:
            # if epoch == 0 and step == 0:
            #    self.__save_checkpoint(model, epoch)

    def test_cnn(self, plot_confusion_matrix: bool = True) -> list[int]:
        model = get_cnn_model(self.params).to(self.device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # Load saved model weights
        self.__load_model_weights(model)

        total_samples = 0
        total_loss = 0.0
        correct_predictions = 0

        # Initialize containers for labels and predictions
        all_labels = []
        all_predictions = []

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

                # Append for confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

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

        if plot_confusion_matrix:
            self.__plot_confusion_matrix(all_labels, all_predictions)
        return all_labels, all_predictions

    def __load_model_weights(self, model: nn.Module):
        if not self.params.model_weights:
            return
        path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            self.params.model_weights,
        )
        model.load_state_dict(torch.load(path))
        logging.info(f"Loaded model weights from {self.params.model_weights}")

    def __get_scheduler(self, optimizer: optim.Optimizer) -> LambdaLR:
        def lr_lambda(step: int) -> float:
            if self.params.lr_mode == "progressive_drops":
                if self.current_epoch >= int(0.75 * self.params.cnn_epochs):
                    return 0.01
                return (
                    0.1
                    if self.current_epoch >= int(0.45 * self.params.cnn_epochs)
                    else 1
                )

        def lr_lambda_for_fine_tuning(step: int) -> float:
            if self.params.lr_mode == "progressive_drops":
                if self.current_epoch >= int(0.75 * self.params.cnn_epochs):
                    return 0.01
                return (
                    0.1
                    if self.current_epoch >= int(0.15 * self.params.cnn_epochs)
                    else 1
                )

        if self.params.fine_tunning:
            return LambdaLR(optimizer, lr_lambda=lr_lambda_for_fine_tuning)
        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def __get_criterion(self) -> nn.Module:
        criterion = nn.CrossEntropyLoss()
        if self.params.class_weights:
            class_weights = self.dataset.get_class_weights_tensor().to(self.device)
            logging.info(f"Using class weights: {class_weights}")
            criterion.weight = class_weights
        return criterion

    def __plot_confusion_matrix(self, all_labels: list, all_predictions: list):
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.params.class_names,
            yticklabels=self.params.class_names,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def __save_checkpoint(self, model: nn.Module, epoch: int):
        os.makedirs(self.params.directories["cnn_checkpoint_weights"], exist_ok=True)
        logging.info(f"Saving checkpoint for epoch {epoch}")
        checkpoint_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.params.algorithm}_test_checkpoint_epoch_{epoch}.pth",
        )
        torch.save(model.state_dict(), checkpoint_path)

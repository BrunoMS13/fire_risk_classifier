import os
import json
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from fire_risk_classifier.dataclasses.params import Params
from fire_risk_classifier.classifier.cnn import get_cnn_model
from fire_risk_classifier.data.image_dataset import CustomImageDataset


class Pipeline:
    def __init__(self, params: Params = Params(), args: dict | None = None):
        """
        Initialize baseline class, prepare data, and calculate class weights.
        :param params: global parameters, used to find location of the dataset and json file
        :return:
        """
        self.params = params

        if args is None:
            args = {}
        if args.get("train"):
            self.params.train_cnn = True
        if args.get("test"):
            self.params.test_cnn = True
        if args.get("algorithm"):
            self.params.algorithm = args.get("algorithm")
        if args.get("calculate_ndvi_index"):
            self.params.calculate_ndvi_index = True
        if args.get("load_weights"):
            self.params.model_weights = args.get("load_weights")
        if args.get("num_epochs"):
            self.params.cnn_epochs = args.get("num_epochs")
        if args.get("batch_size"):
            self.params.batch_size_cnn = args.get("batch_size")
        if args.get("fine_tunning"):
            self.params.fine_tunning = True
        if args.get("class_weights"):
            self.params.class_weights = args.get("class_weights")
        if args.get("images_dir"):
            self.params.directories["images_directory"] = args.get("images_dir")
        if args.get("save_as"):
            self.params.save_as = args.get("save_as")
        if args.get("num_classes"):
            num_classes = args.get("num_classes")
            self.params.num_labels = num_classes
            path = "fire_risk_classifier/data/csvs"
            self.params.directories["annotations_file"] = (
                f"{path}/train_{num_classes}classes.csv"
            )
            self.params.directories["testing_annotations_file"] = (
                f"{path}/test_{num_classes}classes.csv"
            )

        self.__init_data_loaders()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

    # ------------------- Data Loaders ------------------- #

    def __init_data_loaders(self):
        directories = self.params.directories
        batch_size = self.params.batch_size_cnn

        if self.params.train_cnn:
            self.__init_train_dataloader(directories, batch_size)

        if self.params.test_cnn:
            self.__init_test_dataloader(directories, batch_size)

    def __init_train_dataloader(self, directories: dict[str, str], batch_size: int):
        if self.params.train_cnn:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ToTensor(),
                ]
            )

            # Training data loader
            self.dataset = CustomImageDataset(
                directories["images_directory"],
                directories["annotations_file"],
                transform=transform,
                ndvi_index=self.params.calculate_ndvi_index,
                normalize_transform=self.__get_normalize_transform(),
            )
            self.data_loader = DataLoader(
                self.dataset, batch_size=batch_size, shuffle=True
            )

    def __init_test_dataloader(self, directories: dict[str, str], batch_size: int):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # Testing data loader
        self.test_dataset = CustomImageDataset(
            directories["images_directory"],
            directories["testing_annotations_file"],
            transform=transform,
            ndvi_index=self.params.calculate_ndvi_index,
            normalize_transform=self.__get_normalize_transform(),
        )
        self.test_data_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )

    def __get_normalize_transform(self) -> transforms.Normalize:
        return (
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5, 0.0),
                std=(0.5, 0.5, 0.5, 1.0),
            )
            if self.params.calculate_ndvi_index
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )

    # ------------------- Training and Testing ------------------- #

    def train_cnn(self):
        model = get_cnn_model(self.params).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.current_epoch = 0

        criterion = self.__get_criterion()
        scheduler = self.__get_scheduler(optimizer)

        # Load saved model weights
        self.__load_model_weights(model)
        epoch_data = {"loss": [], "accuracy": []}

        for epoch in range(self.params.cnn_epochs):
            self.current_epoch = epoch
            loss, accuracy = self.__training_step(
                epoch, model, optimizer, scheduler, criterion
            )
            epoch_data["loss"].append(loss)
            epoch_data["accuracy"].append(accuracy)
            torch.cuda.empty_cache()

        self.__save_model(model, epoch_data)

    def __save_model(self, model: nn.Module, epoch_data: dict):
        # Save model
        logging.info(
            f"Saving final model to {self.params.directories['cnn_checkpoint_weights']}"
        )
        model_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.__create_model_name()}.pth",
        )
        os.makedirs(self.params.directories["cnn_checkpoint_weights"], exist_ok=True)
        torch.save(model.state_dict(), model_path)

        # Save metrics
        metrics_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.__create_model_name()}_metrics.json",
        )
        with open(metrics_path, "w") as f:
            json.dump(epoch_data, f)

    def __create_model_name(self) -> str:
        if self.params.save_as:
            return self.params.save_as
        if self.__is_body():
            return f"{self.params.algorithm}_body_2C"
        fine_tunned = "FT" if self.params.fine_tunning else "NFT"
        class_weights = "CW" if self.params.class_weights else "NCW"
        return f"{self.params.algorithm}_{class_weights}_{fine_tunned}_final_model"

    def __is_body(self) -> bool:
        return self.params.cnn_epochs == 12

    def __training_step(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LambdaLR,
        criterion: nn.Module,
    ) -> tuple:
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

        avg_loss = running_loss / total_steps
        accuracy = 100 * correct_predictions / total_samples
        return avg_loss, accuracy

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
        model.load_state_dict(torch.load(path, weights_only=False))
        logging.info(f"Loaded model weights from {path}")

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

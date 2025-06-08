import os
import json
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import LambdaLR

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from fire_risk_classifier.dataclasses.params import Params
from fire_risk_classifier.classifier.cnn import get_cnn_model
from fire_risk_classifier.data.image_dataset import CustomImageDataset
from fire_risk_classifier.scripts.unfreeze_layers import UnfreezeLayers


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
        if args.get("ndvi"):
            self.params.calculate_ndvi_index = True
        if args.get("rgbi"):
            self.params.calculate_rgbi = True
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
        if args.get("lr"):
            self.params.cnn_adam_learning_rate = args.get("lr")
        if args.get("wd"):
            self.params.cnn_adam_weight_decay = args.get("wd")
        if args.get("unfreeze"):
            self.params.unfreeze_strategy = args.get("unfreeze")

        self.__init_data_loaders()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        logging.info(f"Images directory: {self.params.directories['images_directory']}")

    # ------------------- Data Loaders ------------------- #

    def __init_data_loaders(self):
        directories = self.params.directories
        batch_size = self.params.batch_size_cnn

        if self.params.train_cnn:
            self.__init_train_and_val_dataloaders(directories, batch_size)

        if self.params.test_cnn:
            self.__init_test_dataloader(directories, batch_size)

    def __init_train_and_val_dataloaders(
        self, directories: dict[str, str], batch_size: int
    ):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
            ]
        )

        # Training data loader
        self.dataset = CustomImageDataset(
            directories["images_directory"],
            directories["annotations_file"],
            transform=transform,
            ndvi_index=self.params.calculate_ndvi_index,
            rgbi=self.params.calculate_rgbi,
        )
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        generator = torch.Generator().manual_seed(42)

        train_dataset, val_dataset = random_split(
            self.dataset, [train_size, val_size], generator=generator
        )

        self.data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_data_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    def __init_test_dataloader(self, directories: dict[str, str], batch_size: int):
        # Testing data loader
        self.test_dataset = CustomImageDataset(
            directories["images_directory"],
            directories["testing_annotations_file"],
            ndvi_index=self.params.calculate_ndvi_index,
            rgbi=self.params.calculate_rgbi,
        )
        self.test_data_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )


    # ------------------- Training and Testing ------------------- #

    def train_cnn(self):
        model = get_cnn_model(self.params).to(self.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.params.cnn_adam_learning_rate,
            weight_decay=self.params.cnn_adam_weight_decay,
        )
        logging.info(f"LR: {self.params.cnn_adam_learning_rate} | WD: {self.params.cnn_adam_weight_decay} | Unfreeze: {self.params.unfreeze_strategy}")
        self.current_epoch = 0

        criterion = self.__get_criterion()
        scheduler = self.__get_scheduler(optimizer)

        best_val_loss = 1
        early_stop_counter = 0

        # Load saved model weights
        self.__load_model_weights(model)
        epoch_data = self.__get_init_epoch_data()

        temp_best_model = None

        for epoch in range(self.params.cnn_epochs):
            logging.info(
                f"Start of Epoch Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

            self.current_epoch = epoch

            if self.params.unfreeze_strategy == "Gradual":
                UnfreezeLayers.unfreeze_layers(model, epoch, self.params)

            loss, accuracy = self.__training_step(
                epoch, model, optimizer, scheduler, criterion
            )
            torch.cuda.empty_cache()
            val_loss, val_accuracy = self.__validation_step(model, criterion)
            torch.cuda.empty_cache()

            logging.info(
                f"End of Epoch Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB"
            )

            self.__append_epoch_data(epoch_data, loss, accuracy, val_loss, val_accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                temp_best_model = model
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.params.patience:
                logging.info(
                    f"Early stopping at Epoch {epoch+1} (Best Val Loss: {best_val_loss:.2f}%)"
                )
                break

        self.__save_model(temp_best_model, epoch_data)

    def __save_model(self, model: nn.Module, epoch_data: dict, extra_info: str = ""):
        # Save model
        logging.info(
            f"Saving final model to {self.params.directories['cnn_checkpoint_weights']}"
        )
        model_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.__create_model_name(extra_info)}.pth",
        )
        os.makedirs(self.params.directories["cnn_checkpoint_weights"], exist_ok=True)
        torch.save(model.state_dict(), model_path)

        # Save metrics
        metrics_path = os.path.join(
            self.params.directories["cnn_checkpoint_weights"],
            f"{self.__create_model_name(extra_info)}_metrics.json",
        )
        if not epoch_data:
            return
        with open(metrics_path, "w") as f:
            json.dump(epoch_data, f)

    def __create_model_name(self, extra_info: str) -> str:
        if self.params.save_as:
            return self.params.save_as + extra_info
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
    ) -> tuple[float, float]:
        model.train()

        total_samples = 0
        running_loss = 0.0
        correct_predictions = 0
        total_steps = len(self.data_loader)

        all_labels = []
        all_predictions = []
        for step, (images, labels) in enumerate(self.data_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            outputs = model(images)
            if self.params.num_labels == 2:
                loss = criterion(outputs, labels.unsqueeze(1).float())
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            with torch.no_grad():
                if self.params.num_labels > 2:
                    predicted = torch.argmax(outputs, dim=1)
                else:
                    predicted = (torch.sigmoid(outputs) >= 0.5).float()

                correct_predictions += (
                    (predicted.view(-1) == labels.view(-1)).sum().item()
                )
                total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if step % 5 == 0 or step == total_steps - 1:
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

    def test_cnn(
        self, plot_confusion_matrix: bool = True, log_info: bool = True
    ) -> tuple:
        model = get_cnn_model(self.params).to(self.device)
        model.train(mode=False)
        criterion = self.__get_criterion()

        # Load saved model weights
        self.__load_model_weights(model)

        total_samples = 0
        total_loss = 0.0
        correct_predictions = 0

        # Initialize containers for labels and predictions
        all_labels = []
        all_predictions = []

        import time
        init_time = time.time()

        with torch.no_grad():  # Disable gradient computation for testing
            for step, (images, labels) in enumerate(
                self.test_data_loader
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                if self.params.num_labels == 2:
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                else:
                    loss = criterion(outputs, labels)
                total_loss += loss.item()

                if self.params.num_labels > 2:
                    predicted = torch.argmax(outputs, dim=1)
                else:
                    predicted = (
                        torch.sigmoid(outputs) >= 0.5
                    ).float()

                correct_predictions += (
                    (predicted.view(-1) == labels.view(-1)).sum().item()
                )
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(
                    predicted.view(-1).cpu().numpy().astype(int).tolist()
                )

                """if step % 10 == 0 and log_info:
                    logging.info(
                        f"Test Step [{step + 1}/{len(self.test_data_loader)}], "
                        f"Loss: {total_loss / (step + 1):.4f}, "
                        f"Accuracy: {100 * correct_predictions / total_samples:.2f}%"
                    )"""
        print(f"Time: {time.time() - init_time}")

        final_loss = total_loss / len(self.test_data_loader)
        final_accuracy = 100 * correct_predictions / total_samples
        f1 = f1_score(all_labels, all_predictions, average="binary" if self.params.num_labels == 2 else "macro")

        logging.info(f"Final Test Loss: {final_loss:.4f} | Final Test Accuracy: {final_accuracy:.2f}% | Final Test F1: {f1:.4f}")

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
            return 1.0

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def __get_criterion(self) -> nn.Module:
        if self.params.class_weights:
            class_weights = self.test_dataset.get_class_weights_tensor().to(self.device)
            logging.info(f"Using class weights: {class_weights}")

            if self.params.num_labels > 2:
                return nn.CrossEntropyLoss(weight=class_weights)
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)

        return (
            nn.CrossEntropyLoss()
            if self.params.num_labels > 2
            else nn.BCEWithLogitsLoss()
        )

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

    def __validation_step(self, model: nn.Module, criterion: nn.Module) -> tuple[float, float]:
        model.eval()  # Set model to evaluation mode

        total_samples = 0
        running_loss = 0.0
        correct_predictions = 0

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in self.val_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                if self.params.num_labels == 2:
                    loss = criterion(outputs, labels.unsqueeze(1).float())
                else:
                    loss = criterion(outputs, labels)
                running_loss += loss.item()

                predicted = (
                    torch.argmax(outputs, dim=1)
                    if self.params.num_labels > 2
                    else (torch.sigmoid(outputs) >= 0.5).float()
                )
                correct_predictions += (
                    (predicted.view(-1) == labels.view(-1)).sum().item()
                )
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = running_loss / len(self.val_data_loader)
        accuracy = 100 * correct_predictions / total_samples
        logging.info(
            f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {accuracy:.2f}%"
        )
        return avg_loss, accuracy

    def __get_init_epoch_data(self) -> dict:
        return {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
    
    def __append_epoch_data(self, epoch_data: dict[str, list], loss: float, accuracy: float, val_loss: float, val_acc: float):
        epoch_data["train_loss"].append(loss)
        epoch_data["train_accuracy"].append(accuracy)
        epoch_data["val_loss"].append(val_loss)
        epoch_data["val_accuracy"].append(val_acc)

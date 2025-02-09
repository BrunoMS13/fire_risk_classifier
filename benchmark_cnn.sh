#!/bin/bash

echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed"

NUM_CLASSES=2
mkdir -p ~/models

docker build -t fire_risk_classifier_image .

# Train EfficientNet B5 with optimized parameters
docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b5 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 5e-4 --lr 2e-4 --save_as eb5_rgb_wd5e4_lr2e4

echo "Copying EfficientNet B5..."

cp -r $WEIGHTS_PATH/eb5_rgb_wd5e4_lr2e4.pth ~/models
cp -r $WEIGHTS_PATH/eb5_rgb_wd5e4_lr2e4_metrics.json ~/models

# Train DenseNet161 with optimized parameters
docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet161 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 6e-4 --lr 4e-4 --save_as d161_rgb_wd6e4_lr4e4

echo "Copying DenseNet161..."

cp -r $WEIGHTS_PATH/d161_rgb_wd6e4_lr4e4.pth ~/models
cp -r $WEIGHTS_PATH/d161_rgb_wd6e4_lr4e4_metrics.json ~/models

# Train ResNet101 with optimized parameters
docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet101 --batch_size 16 --train True --num_epochs 18 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 6e-4 --lr 4.5e-4 --save_as r101_rgb_wd6e4_lr4p5e4

echo "Copying ResNet101..."

cp -r $WEIGHTS_PATH/r101_rgb_wd6e4_lr4p5e4.pth ~/models
cp -r $WEIGHTS_PATH/r101_rgb_wd6e4_lr4p5e4_metrics.json ~/models

echo "Benchmark trainings complete!"

docker rmi fire_risk_classifier_image

#!/bin/bash

echo "Building Docker Image..."
docker build -t fire_risk_classifier_image .

# Paths
WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"
NUM_CLASSES=2
mkdir -p ~/models

# Define hyperparameters to test
LEARNING_RATES=("1e-6" "5e-6" "5e-5" "1e-5")  # Expanded learning rate options
WEIGHT_DECAY="5e-4"  # Fixed weight decay for all runs
UNFREEZING=("nothing")

# Logging file
LOG_FILE=~/models/training_results.log
echo "Experiment Results - $(date)" > $LOG_FILE

# Loop through learning rates and unfreezing strategies
for lr in "${LEARNING_RATES[@]}"; do
    for unfreeze in "${UNFREEZING[@]}"; do
        
        EXP_NAME="r50_irg_wd${WEIGHT_DECAY}_lr${lr}_patience8"

        echo "Training ResNet50 with lr=$lr, wd=$WEIGHT_DECAY, unfreeze=$unfreeze"
        docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train \
            --algorithm resnet50 --batch_size 16 --train True --num_epochs 18 --num_classes $NUM_CLASSES \
            --images_dir $IMAGE_DIR --wd $WEIGHT_DECAY --lr $lr --unfreeze $unfreeze --save_as $EXP_NAME

        echo "Copying Results for $EXP_NAME..."
        cp -r $WEIGHTS_PATH/$EXP_NAME.pth ~/models
        cp -r $WEIGHTS_PATH/${EXP_NAME}_metrics.json ~/models

        # Log experiment results
        echo "$EXP_NAME" >> $LOG_FILE
        echo "Learning Rate: $lr, Weight Decay: $WEIGHT_DECAY, Unfreezing: $unfreeze" >> $LOG_FILE
        echo "------------------------------------" >> $LOG_FILE

    done
done

echo "Benchmark trainings complete! Results saved in $LOG_FILE"

# Cleanup Docker Image
docker rmi fire_risk_classifier_image


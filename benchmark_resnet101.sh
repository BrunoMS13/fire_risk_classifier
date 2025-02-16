#!/bin/bash

echo "Building Docker Image..."
docker build -t fire_risk_classifier_image .

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"
NUM_CLASSES=2
mkdir -p ~/models

# Define hyperparameters to test
LEARNING_RATES=("1e-4" "5e-4" "1e-3")
WEIGHT_DECAYS=("1e-4" "5e-4" "1e-3")
UNFREEZING=("Gradual" "Nothing")

# Logging file
LOG_FILE=~/models/training_results.log
echo "Experiment Results - $(date)" > $LOG_FILE

# Loop through different hyperparameter combinations
for lr in "${LEARNING_RATES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
        for unfreeze in "${UNFREEZING[@]}"; do
            
            EXP_NAME="r101_irg_wd${wd}_lr${lr}_unfreeze${unfreeze}"

            echo "Training ResNet101 with lr=$lr, wd=$wd, unfreeze=$unfreeze"
            docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train \
                --algorithm resnet101 --batch_size 16 --train True --num_epochs 18 --num_classes $NUM_CLASSES \
                --images_dir $IMAGE_DIR --wd $wd --lr $lr --unfreeze $unfreeze --save_as $EXP_NAME

            echo "Copying Results for $EXP_NAME..."
            cp -r $WEIGHTS_PATH/$EXP_NAME.pth ~/models
            cp -r $WEIGHTS_PATH/${EXP_NAME}_metrics.json ~/models

            # Log experiment results
            echo "$EXP_NAME" >> $LOG_FILE
            echo "Learning Rate: $lr, Weight Decay: $wd, Unfreezing: $unfreeze" >> $LOG_FILE
            echo "------------------------------------" >> $LOG_FILE

        done
    done
done

echo "Benchmark trainings complete! Results saved in $LOG_FILE"

# Cleanup
docker rmi fire_risk_classifier_image

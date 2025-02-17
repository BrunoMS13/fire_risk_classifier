#!/bin/bash

cd /tmp

git clone https://github.com/BrunoMS13/fire_risk_classifier.git

mkdir -p fire_risk_classifier/fire_risk_classifier/data/images && \
cp -r ~/fire_risk_classifier/fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed fire_risk_classifier/fire_risk_classifier/data/images/

cd fire_risk_classifier

# Check if the Docker image exists before trying to remove it
if docker image inspect fire_risk_classifier_image > /dev/null 2>&1; then
    echo "Removing existing Docker image..."
    docker rmi fire_risk_classifier_image
else
    echo "No existing Docker image found. Skipping removal."
fi

echo "Building Docker Image..."
docker build -t fire_risk_classifier_image .

# Paths
WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"
NUM_CLASSES=2
mkdir -p ~/models

# Define hyperparameters to test
LEARNING_RATES=("1e-4" "1e-5")  # Only these LRs for new models
WEIGHT_DECAY="0"  # Fixed weight decay for all runs
UNFREEZING=("nothing")

# Logging file
LOG_FILE=~/models/training_results.log
echo "Experiment Results - $(date)" > $LOG_FILE

# Model architectures to train
MODELS=("densenet161")

# Number of runs per configuration
NUM_RUNS=2

# Loop through models, learning rates, and unfreezing strategies
for model in "${MODELS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for unfreeze in "${UNFREEZING[@]}"; do
            for run in $(seq 1 $NUM_RUNS); do
                
                EXP_NAME="${model}_irg_wd${WEIGHT_DECAY}_lr${lr}_run${run}"

                echo "Training $model with lr=$lr, wd=$WEIGHT_DECAY, unfreeze=$unfreeze (Run $run)"
                docker run --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train \
                    --algorithm $model --batch_size 16 --train True --num_epochs 50 --num_classes $NUM_CLASSES \
                    --images_dir $IMAGE_DIR --wd $WEIGHT_DECAY --lr $lr --unfreeze $unfreeze --save_as $EXP_NAME

                echo "Copying Results for $EXP_NAME..."
                cp -r $WEIGHTS_PATH/$EXP_NAME.pth ~/models
                cp -r $WEIGHTS_PATH/${EXP_NAME}_metrics.json ~/models

                # Log experiment results
                echo "$EXP_NAME" >> $LOG_FILE
                echo "Model: $model, Learning Rate: $lr, Weight Decay: $WEIGHT_DECAY, Unfreezing: $unfreeze, Run: $run" >> $LOG_FILE
                echo "------------------------------------" >> $LOG_FILE

            done
        done
    done
done

echo "Benchmark trainings complete! Results saved in $LOG_FILE"

# Cleanup Docker Image
docker rmi fire_risk_classifier_image

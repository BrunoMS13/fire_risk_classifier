#!/bin/bash

cd /tmp

git clone https://github.com/BrunoMS13/fire_risk_classifier.git

# Create image directory and copy both IRG and RGB images
mkdir -p fire_risk_classifier/fire_risk_classifier/data/images
cp -r ~/fire_risk_classifier/fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed fire_risk_classifier/fire_risk_classifier/data/images/
cp -r ~/fire_risk_classifier/fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed fire_risk_classifier/fire_risk_classifier/data/images/

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
IMAGE_BASE_DIR="fire_risk_classifier/data/images"
NUM_CLASSES=3
mkdir -p ~/models

# Define learning rates to test
LEARNING_RATES=("1e-4")  # Keeping only two learning rates

# Define weight decay values to test
WEIGHT_DECAYS=("1e-4")  # Removed 1e-6

# Define unfreezing strategies (kept only "Gradual")
UNFREEZE_OPTIONS=("None")

# Logging file (add INSTANCE_ID to log for different shells)
INSTANCE_ID=${INSTANCE_ID:-"instance_2"}  # Default to "instance_1" if not set
LOG_FILE=~/models/training_results_${INSTANCE_ID}.log
echo "Experiment Results - $(date)" > $LOG_FILE

# Model architectures to train
MODELS=("densenet161")

# Number of runs per configuration
NUM_RUNS=1

# Define datasets
DATASETS=(
    "RGB fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed"
    "IRG fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"
    "RGB_NDVI fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed --ndvi True"
)


# Loop through models, datasets, learning rates, weight decay values, and unfreezing strategies
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        DATASET_NAME=$(echo "$dataset" | awk '{print $1}')
        IMAGE_DIR=$(echo "$dataset" | awk '{print $2}')
        NDVI_FLAG=$(echo "$dataset" | awk '{print $3 " " $4}')

        for lr in "${LEARNING_RATES[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
                for unfreeze in "${UNFREEZE_OPTIONS[@]}"; do
                    for run in $(seq 1 $NUM_RUNS); do
                        EXP_NAME="${model}_${DATASET_NAME}_lr${lr}_wd${wd}_unfreeze${unfreeze}_run${run}_${INSTANCE_ID}"
                        echo "Training $model on $DATASET_NAME with lr=$lr, wd=$wd, unfreeze=$unfreeze (Run $run) $NDVI_FLAG"

                        docker run --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train \
                            --algorithm $model --batch_size 8 --train True --num_epochs 1 --num_classes $NUM_CLASSES \
                            --images_dir $IMAGE_DIR --wd $wd --lr $lr --unfreeze $unfreeze --save_as $EXP_NAME $NDVI_FLAG

                        echo "Copying Results for $EXP_NAME..."
                        cp -r $WEIGHTS_PATH/$EXP_NAME.pth ~/models
                        cp -r $WEIGHTS_PATH/${EXP_NAME}_metrics.json ~/models

                        # Log experiment results
                        echo "$EXP_NAME" >> $LOG_FILE
                        echo "Model: $model, Dataset: $DATASET_NAME, Learning Rate: $lr, Weight Decay: $wd, Unfreezing: $unfreeze, Run: $run" >> $LOG_FILE
                        echo "------------------------------------" >> $LOG_FILE

                    done
                done
            done
        done
    done
done

echo "Benchmark trainings complete! Results saved in $LOG_FILE"

# Cleanup Docker Image
docker rmi fire_risk_classifier_image

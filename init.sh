#!/bin/bash

# Exit immediately if a command fails
set -e

# Set dataset type (Change to "IRG" if needed)
DATASET_TYPE="IRG"  # Change this to "IRG" if needed

# Define dataset folder based on type
DATASET_FOLDER="ortos2018-${DATASET_TYPE}-62_5m-decompressed"

# Navigate to the appropriate directory
cd ~/fire_risk_classifier/fire_risk_classifier/data || { echo "Error: Directory not found!"; exit 1; }

# Create the images directory if it doesn't exist
mkdir -p images

# Check if the source directory exists and copy it into images/
if [ -d ~/fire_risk_classifier/fire_risk_classifier/data/images/$DATASET_FOLDER ]; then
    echo "Copying $DATASET_FOLDER into images/ directory..."
    cp -r ~/fire_risk_classifier/fire_risk_classifier/data/images/$DATASET_FOLDER images/
    echo "Files copied successfully!"
else
    echo "Error: Source folder '$DATASET_FOLDER' does not exist!"
    exit 1
fi

echo "Setup completed successfully!"

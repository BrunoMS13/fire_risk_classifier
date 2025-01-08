echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

docker build -t fire_risk_classifier_image .

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet --batch_size 32 --train True --ndvi True

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet --batch_size 64 --train True --ndvi True

cp -r $WEIGHTS_PATH/* ~/models

# Build the Docker image
docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --ndvi True --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet_body_2C.pth

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --ndvi True --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet_body_2C.pth

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet --num_epochs 5 --batch_size 32 --train True --ndvi True --class_weights class_weights --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/resnet_body_2C.pth

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet --num_epochs 5 --batch_size 32 --train True --ndvi True --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/resnet_body_2C.pth

echo "Copying models to ~/models/..."

cp -r $WEIGHTS_PATH/* ~/models/2C_NDVI
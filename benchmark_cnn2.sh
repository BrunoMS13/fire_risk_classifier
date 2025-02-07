echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed"

NUM_CLASSES=2
mkdir -p ~/models/2C

docker build -t fire_risk_classifier_image .

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b5 --batch_size 16 --train True --num_epochs 12 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 3e-4

echo "Copying efficientnet_b5..."

mkdir -p ~/models/2C/efficientnet_b5
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C.pth ~/models/2C/efficientnet_b5
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C_metrics.json ~/models/2C/efficientnet_b5

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet161 --batch_size 16 --train True --num_epochs 12 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 3e-4

echo "Copying densenet161..."

mkdir -p ~/models/2C/densenet161_
cp -r $WEIGHTS_PATH/densenet161_body_2C.pth ~/models/2C/densenet161
cp -r $WEIGHTS_PATH/densenet161_body_2C_metrics.json ~/models/2C/densenet161

echo "Benchmark trainings complete!"

docker rmi fire_risk_classifier_image

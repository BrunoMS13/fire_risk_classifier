echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"

NUM_CLASSES=2
mkdir -p ~/models

docker build -t fire_risk_classifier_image .

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b5 --batch_size 16 --train True --num_epochs 12 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 1e-5 --lr 1e-4

echo "Copying efficientnet_b5..."

mkdir -p ~/models/efficientnet_b5_wd1e5_lr1e4
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C.pth ~/models/efficientnet_b5_wd1e5_lr1e4
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C_metrics.json ~/models/efficientnet_b5_wd1e5_lr1e4

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet161 --batch_size 16 --train True --num_epochs 12 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 1e-5 --lr 1e-4

echo "Copying densenet161..."

mkdir -p ~/models/densenet161_wd1e5_lr1e4
cp -r $WEIGHTS_PATH/densenet161_body_2C.pth ~/models/densenet161_wd1e5_lr1e4
cp -r $WEIGHTS_PATH/densenet161_body_2C_metrics.json ~/models/densenet161_wd1e5_lr1e4

#resnet101

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet101 --batch_size 16 --train True --num_epochs 12 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 1e-5 --lr 1e-4

echo "Copying resnet101..."

mkdir -p ~/models/resnet101_wd1e5_lr1e4
cp -r $WEIGHTS_PATH/resnet101_body_2C.pth ~/models/resnet101_wd1e5_lr1e4
cp -r $WEIGHTS_PATH/resnet101_body_2C_metrics.json ~/models/resnet101_wd1e5_lr1e4

echo "Benchmark trainings complete!"

docker rmi fire_risk_classifier_image

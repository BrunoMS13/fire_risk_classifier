echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-IRG-62_5m-decompressed"

NUM_CLASSES=2
mkdir -p ~/models

docker build -t fire_risk_classifier_image .

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b5 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 2e-4 --lr 3e-4 --save_as eb5_wd2e4_lr3e4

echo "Copying efficientnet_b5..."

mkdir -p ~/models/eb5_wd2e4_lr3e4
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C.pth ~/models/eb5_wd2e4_lr3e4
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C_metrics.json ~/models/eb5_wd2e4_lr3e4

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet161 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 5e-4 --lr 5e-4 --save_as d161_wd5e4_lr5e4

echo "Copying densenet161..."

mkdir -p ~/models/d161_wd5e4_lr5e4
cp -r $WEIGHTS_PATH/densenet161_body_2C.pth ~/models/d161_wd5e4_lr5e4
cp -r $WEIGHTS_PATH/densenet161_body_2C_metrics.json ~/models/d161_wd5e4_lr5e4

#resnet101

RESNET_LR="5e-4"
RESNET_WD="5e-4"
SAVE_AS_RESNET101="r101_wd5e-4_lr5e-4"
docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet101 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 5e-4 --lr 5e-4 --save_as r101_wd5e4_lr5e4

echo "Copying resnet101..."

mkdir -p ~/models/r101_wd5e4_lr5e4
cp -r $WEIGHTS_PATH/resnet101_body_2C.pth ~/models/r101_wd5e4_lr5e4
cp -r $WEIGHTS_PATH/resnet101_body_2C_metrics.json ~/models/r101_wd5e4_lr5e4

echo "Benchmark trainings complete!"

docker rmi fire_risk_classifier_image

echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"
IMAGE_DIR="fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed"

NUM_CLASSES=2
mkdir -p ~/models

docker build -t fire_risk_classifier_image .

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b5 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 3e-4 --lr 3e-4 --save_as eb5_rgb_wd3e4_lr3e4

echo "Copying efficientnet_b5..."

cp -r $WEIGHTS_PATH/eb5_rgb_wd3e4_lr3e4.pth ~/models
cp -r $WEIGHTS_PATH/eb5_rgb_wd3e4_lr3e4_metrics.json ~/models

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet161 --batch_size 16 --train True --num_epochs 15 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 5e-4 --lr 5e-4 --save_as d161_rgb_wd5e4_lr5e4

echo "Copying densenet161..."

cp -r $WEIGHTS_PATH/d161_rgb_wd5e4_lr5e4.pth ~/models
cp -r $WEIGHTS_PATH/d161_rgb_wd5e4_lr5e4_metrics.json ~/models

docker run -it --rm --gpus all -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet101 --batch_size 16 --train True --num_epochs 18 --num_classes $NUM_CLASSES --images_dir $IMAGE_DIR --wd 5e-4 --lr 5e-4 --save_as r101_rgb_wd5e4_lr5e4

echo "Copying resnet101..."

cp -r $WEIGHTS_PATH/r101_rgb_wd5e4_lr5e4.pth ~/models
cp -r $WEIGHTS_PATH/r101_rgb_wd5e4_lr5e4_metrics.json ~/models

echo "Benchmark trainings complete!"

docker rmi fire_risk_classifier_image

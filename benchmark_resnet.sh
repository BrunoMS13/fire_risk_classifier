echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

docker build -t fire_risk_classifier_image .

echo "Benchmark training resnets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet --batch_size 64 --train True --num_epochs 12

echo "Copying resnet50..."

mkdir -p ~/models/resnet50
cp -r $WEIGHTS_PATH/resnet_body_2C.pth ~/models/resnet50
cp -r $WEIGHTS_PATH/resnet_body_2C_metrics.json ~/models/resnet50

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet101 --batch_size 64 --train True --num_epochs 12

echo "Copying resnet101..."

mkdir -p ~/models/resnet101
cp -r $WEIGHTS_PATH/resnet101_body_2C.pth ~/models/resnet101
cp -r $WEIGHTS_PATH/resnet101_body_2C_metrics.json ~/models/resnet101

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet152 --batch_size 64 --train True --num_epochs 12

echo "Copying resnet152..."

mkdir -p ~/models/resnet152
cp -r $WEIGHTS_PATH/resnet152_body_2C.pth ~/models/resnet152
cp -r $WEIGHTS_PATH/resnet152_body_2C_metrics.json ~/models/resnet152

echo "Benchmark training resnets... Done"


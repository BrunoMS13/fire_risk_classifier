echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

docker build -t fire_risk_classifier_image .

echo "Benchmark training densenets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet --batch_size 32 --train True --num_epochs 12

echo "Copying densenet161..."

mkdir -p ~/models/densenet161
cp -r $WEIGHTS_PATH/* ~/models/densenet161

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet169 --batch_size 32 --train True --num_epochs 12

echo "Copying densenet169..."

mkdir -p ~/models/densenet169
cp -r $WEIGHTS_PATH/* ~/models/densenet169

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet201 --batch_size 32 --train True --num_epochs 12

echo "Copying densenet201..."

mkdir -p ~/models/resnet
cp -r $WEIGHTS_PATH/* ~/models/densenet201


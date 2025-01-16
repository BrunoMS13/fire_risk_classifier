echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

docker build -t fire_risk_classifier_image .

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm inception --batch_size 64 --train True

echo "Copying models to ~/models/..."

cp -r $WEIGHTS_PATH/* ~/models

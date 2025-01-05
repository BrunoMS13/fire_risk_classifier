echo "Building Image..."

# Build the Docker image
docker build -t fire_risk_classifier_image .

echo "Running densenet..."

docker run -it --rm -v "/tmp/fire_risk_classifier:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --batch_size 32 --train True

cp -r /tmp/fire_risk_classifier/resnet_body_2C.pth ~/rgb_models_new/

echo "Running resnet..."

docker run -it --rm -v "/tmp/fire_risk_classifier:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --batch_size 64 --train True

cp -r /tmp/fire_risk_classifier/densenet_body_2C.pth ~/rgb_models_new/
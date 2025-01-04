echo "Setting up docker image... (Version 1.1)"

# Build the Docker image
docker build -t fire_risk_classifier_image .

# Run the container, mounting the cnn_checkpoint_weights folder
docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --batch_size 32 --train True

docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --batch_size 64 --train True

mv tmp/fire_risk_classifier/data/cnn_checkpoint_weights/* ~/fire_risk_classifier/rgb_models
echo "Building Image..."

# Build the Docker image
docker build -t fire_risk_classifier_image .

# Run the container, mounting the cnn_checkpoint_weights folder
docker run -it --rm -v "tmp/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --batch_size 32 --train True

cp -r tmp/fire_risk_classifier/data/cnn_checkpoint_weights/* ~/models
echo "Setting up docker image..."

# Build and run the Docker container
docker build -t fire_risk_classifier_image .
docker run -it --rm fire_risk_classifier_image

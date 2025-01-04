echo "Setting up docker image... (Version 1.1)"

# Build the Docker image
docker build -t fire_risk_classifier_image .

echo "Training the hydra body..."

# Run the container, mounting the cnn_checkpoint_weights folder
docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --batch_size 32 --train True

docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --batch_size 32 --train True

echo "Training and refining the hydra heads..."

docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/densenet_body_2C.pth

docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/densenet_body_2C.pth

docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/resnet_body_2C.pth

docker run -it --rm -v "$(pwd)/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/resnet_body_2C.pth

mv tmp/fire_risk_classifier/data/cnn_checkpoint_weights/* /fire_risk_classifier/data/cnn_checkpoint_weights/backup
echo "Building Image..."

# Build the Docker image
docker run -it --rm -v "/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/densenet_body_2C.pth

docker run -it --rm -v "/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/densenet_body_2C.pth

docker run -it --rm -v "/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/resnet_body_2C.pth

docker run -it --rm -v "/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights:/app/fire_risk_classifier/data/cnn_checkpoint_weights" fire_risk_classifier_image poetry run train --algorithm resnet --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights /app/fire_risk_classifier/data/cnn_checkpoint_weights/resnet_body_2C.pth

echo "Copying models to ~/models/..."

cp -r /tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights/* ~/models/
WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

cd /tmp

git clone https://github.com/BrunoMS13/fire_risk_classifier.git

mkdir -p fire_risk_classifier/fire_risk_classifier/data/images

echo "Copying images..."

cp -r ~/fire_risk_classifier/fire_risk_classifier/data/images/ortos2018-RGB-62_5m-decompressed \
      /tmp/fire_risk_classifier/fire_risk_classifier/data/images

echo "Copying weights..."

mkdir -p fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights

cp -r ~/models/* \
      /tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights

echo "Building image..."

cd fire_risk_classifier

docker build -t fire_risk_classifier_image .

mkdir -p ~/models/fine_tunned

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet_body_2C.pth

cp -r $WEIGHTS_PATH/densenet_CW_FT_final_model.pth ~/models/fine_tunned
cp -r $WEIGHTS_PATH/densenet_CW_FT_final_model.json ~/models/fine_tunned

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet_body_2C.pth

cp -r $WEIGHTS_PATH/densenet_NCW_FT_final_model.pth ~/models/fine_tunned
cp -r $WEIGHTS_PATH/densenet_NCW_FT_final_model.json ~/models/fine_tunned

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet169 --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet169_body_2C.pth

cp -r $WEIGHTS_PATH/densenet169_CW_FT_final_model.pth ~/models/fine_tunned
cp -r $WEIGHTS_PATH/densenet169_CW_FT_final_model.json ~/models/fine_tunned

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet169 --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet169_body_2C.pth

cp -r $WEIGHTS_PATH/densenet169_NCW_FT_final_model.pth ~/models/fine_tunned
cp -r $WEIGHTS_PATH/densenet169_NCW_FT_final_model.json ~/models/fine_tunned

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet201 --num_epochs 5 --batch_size 32 --train True --class_weights class_weights --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet201_body_2C.pth

cp -r $WEIGHTS_PATH/densenet201_CW_FT_final_model.pth ~/models/fine_tunned
cp -r $WEIGHTS_PATH/densenet201_CW_FT_final_model.json ~/models/fine_tunned

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet201 --num_epochs 5 --batch_size 32 --train True --fine_tunning True --load_weights $DOCKER_WEIGHTS_PATH/densenet201_body_2C.pth

cp -r $WEIGHTS_PATH/densenet201_NCW_FT_final_model.pth ~/models/fine_tunned
cp -r $WEIGHTS_PATH/densenet201_NCW_FT_final_model.json ~/models/fine_tunned

echo "Done..."








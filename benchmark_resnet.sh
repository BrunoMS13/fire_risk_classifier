echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

docker build -t fire_risk_classifier_image .

echo "Benchmark training resnets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet50 --batch_size 64 --train True --num_epochs 12

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

echo "Benchmark training vggs..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm vgg16 --batch_size 64 --train True --num_epochs 12

echo "Copying vgg16..."

mkdir -p ~/models/vgg16
cp -r $WEIGHTS_PATH/vgg_body_2C.pth ~/models/vgg16
cp -r $WEIGHTS_PATH/vgg_body_2C_metrics.json ~/models/vgg16

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm vgg19 --batch_size 64 --train True --num_epochs 12

echo "Copying vgg19..."

mkdir -p ~/models/vgg19
cp -r $WEIGHTS_PATH/vgg19_body_2C.pth ~/models/vgg19
cp -r $WEIGHTS_PATH/vgg19_body_2C_metrics.json ~/models/vgg19

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm vgg19_bn --batch_size 64 --train True --num_epochs 12

echo "Copying vgg19_bn..."

mkdir -p ~/models/vgg19_bn
cp -r $WEIGHTS_PATH/vgg19_bn_body_2C.pth ~/models/vgg19_bn
cp -r $WEIGHTS_PATH/vgg19_bn_body_2C_metrics.json ~/models/vgg19_bn


echo "Benchmark training efficientnets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b4 --batch_size 64 --train True --num_epochs 12

echo "Copying efficientnet_b4..."

mkdir -p ~/models/efficientnet_b4
cp -r $WEIGHTS_PATH/efficientnet_b4_body_2C.pth ~/models/efficientnet_b4
cp -r $WEIGHTS_PATH/efficientnet_b4_body_2C_metrics.json ~/models/efficientnet_b4

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b7 --batch_size 64 --train True --num_epochs 12

echo "Copying efficientnet_b7..."

mkdir -p ~/models/efficientnet_b7
cp -r $WEIGHTS_PATH/efficientnet_b7_body_2C.pth ~/models/efficientnet_b7
cp -r $WEIGHTS_PATH/efficientnet_b7_body_2C_metrics.json ~/models/efficientnet_b7

echo "Benchmark trainings complete!"

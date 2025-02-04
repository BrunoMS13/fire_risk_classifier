echo "Building Image..."

WEIGHTS_PATH="/tmp/fire_risk_classifier/fire_risk_classifier/data/cnn_checkpoint_weights"
DOCKER_WEIGHTS_PATH="/app/fire_risk_classifier/data/cnn_checkpoint_weights"

NUM_CLASSES=2
mkdir -p ~/models/2C

docker build -t fire_risk_classifier_image .

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b4 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying efficientnet_b4..."

mkdir -p ~/models/2C/efficientnet_b4
cp -r $WEIGHTS_PATH/efficientnet_b4_body_2C.pth ~/models/2C/efficientnet_b4
cp -r $WEIGHTS_PATH/efficientnet_b4_body_2C_metrics.json ~/models/2C/efficientnet_b4

: '
echo "Benchmark training resnets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet50 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying resnet50..."

mkdir -p ~/models/2C/resnet50
cp -r $WEIGHTS_PATH/resnet50_body_2C.pth ~/models/3C/resnet50
cp -r $WEIGHTS_PATH/resnet50_body_2C_metrics.json ~/models/3C/resnet50

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet101 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying resnet101..."

mkdir -p ~/models/3C/resnet101
cp -r $WEIGHTS_PATH/resnet101_body_2C.pth ~/models/3C/resnet101
cp -r $WEIGHTS_PATH/resnet101_body_2C_metrics.json ~/models/3C/resnet101

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm resnet152 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying resnet152..."

mkdir -p ~/models/3C/resnet152
cp -r $WEIGHTS_PATH/resnet152_body_2C.pth ~/models/3C/resnet152
cp -r $WEIGHTS_PATH/resnet152_body_2C_metrics.json ~/models/3C/resnet152

echo "Benchmark training vggs..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm vgg19 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying vgg19..."

mkdir -p ~/models/3C/vgg19
cp -r $WEIGHTS_PATH/vgg19_body_2C.pth ~/models/3C/vgg19
cp -r $WEIGHTS_PATH/vgg19_body_2C_metrics.json ~/models/3C/vgg19

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm vgg19_bn --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying vgg19_bn..."

mkdir -p ~/models/3C/vgg19_bn
cp -r $WEIGHTS_PATH/vgg19_bn_body_2C.pth ~/models/3C/vgg19_bn
cp -r $WEIGHTS_PATH/vgg19_bn_body_2C_metrics.json ~/models/3C/vgg19_bn


echo "Benchmark training efficientnets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b4 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying efficientnet_b4..."

mkdir -p ~/models/3C/efficientnet_b4
cp -r $WEIGHTS_PATH/efficientnet_b4_body_2C.pth ~/models/3C/efficientnet_b4
cp -r $WEIGHTS_PATH/efficientnet_b4_body_2C_metrics.json ~/models/3C/efficientnet_b4

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b5 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying efficientnet_b5..."

mkdir -p ~/models/3C/efficientnet_b5
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C.pth ~/models/3C/efficientnet_b5
cp -r $WEIGHTS_PATH/efficientnet_b5_body_2C_metrics.json ~/models/3C/efficientnet_b5

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b6 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying efficientnet_b6..."

mkdir -p ~/models/3C/efficientnet_b6
cp -r $WEIGHTS_PATH/efficientnet_b6_body_2C.pth ~/models/3C/efficientnet_b6
cp -r $WEIGHTS_PATH/efficientnet_b6_body_2C_metrics.json ~/models/3C/efficientnet_b6

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm efficientnet_b7 --batch_size 64 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying efficientnet_b7..."

mkdir -p ~/models/3C/efficientnet_b7
cp -r $WEIGHTS_PATH/efficientnet_b7_body_2C.pth ~/models/3C/efficientnet_b7
cp -r $WEIGHTS_PATH/efficientnet_b7_body_2C_metrics.json ~/models/3C/efficientnet_b7

echo "Benchmark training densenets..."

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet161 --batch_size 32 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying densenet161..."

mkdir -p ~/models/3C/densenet161
cp -r $WEIGHTS_PATH/densenet161_body_2C.pth ~/models/3C/densenet161
cp -r $WEIGHTS_PATH/densenet161_body_2C_metrics.json ~/models/3C/densenet161

docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet169 --batch_size 32 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying densenet169..."

mkdir -p ~/models/3C/densenet169
cp -r $WEIGHTS_PATH/densenet169_body_2C.pth ~/models/3C/densenet169
cp -r $WEIGHTS_PATH/densenet169_body_2C_metrics.json ~/models/3C/densenet169


docker run -it --rm -v "$WEIGHTS_PATH:$DOCKER_WEIGHTS_PATH" fire_risk_classifier_image poetry run train --algorithm densenet201 --batch_size 32 --train True --num_epochs 12 --num_classes $NUM_CLASSES

echo "Copying densenet201..."

mkdir -p ~/models/3C/densenet201
cp -r $WEIGHTS_PATH/densenet201_body_2C.pth ~/models/3C/densenet201
cp -r $WEIGHTS_PATH/densenet201_body_2C_metrics.json ~/models/3C/densenet201
'

echo "Benchmark trainings complete!"

docker rmi fire_risk_classifier_image

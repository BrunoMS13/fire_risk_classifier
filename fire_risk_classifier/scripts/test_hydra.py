"""echo "PREPROCESSING TRAIN DATASET"
https://github.com/maups/hydra-fmow/blob/master/train.sh


python iarpa/runBaseline.py --prepare True --train True

echo "GENERATING MODEL 1/12"
python -u iarpa/runBaseline.py --algorithm resnet50 --train True --num_gpus 4 --num_epochs 12 --batch_size 80 --class_weights no_weights --prefix p01 --generator flip --database v1

echo "GENERATING MODEL 2/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 12 --batch_size 28 --class_weights no_weights --prefix p01 --generator flip --database v1

echo "GENERATING MODEL 3/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 5 --batch_size 28 --class_weights class_pond --prefix p02 --generator flip --fine_tunning True --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v2

echo "GENERATING MODEL 4/12"
python -u iarpa/runBaseline.py --algorithm resnet50 --train True --num_gpus 4 --num_epochs 5 --batch_size 80 --fine_tunning True --class_weights class_pond --prefix p02 --generator flip --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.resnet50.p01.06.hdf5 --database v3

echo "GENERATING MODEL 5/12"
python -u iarpa/runBaseline.py --algorithm resnet50 --train True --num_gpus 4 --num_epochs 5 --batch_size 80 --fine_tunning True --class_weights class_pond --prefix p03 --generator flip --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.resnet50.p01.06.hdf5 --database v2

echo "GENERATING MODEL 6/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 5 --batch_size 28 --class_weights class_pond --prefix p03 --generator flip --fine_tunning True --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v3

echo "GENERATING MODEL 7/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 5 --batch_size 28 --class_weights sklearn_class_weight --prefix p04 --generator zoom --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v1

echo "GENERATING MODEL 8/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 5 --batch_size 28 --class_weights no_weights --prefix p05 --generator shift --fine_tunning True --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v1

echo "GENERATING MODEL 9/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 5 --batch_size 28 --class_weights class_weights --prefix p06 --generator shift --fine_tunning True --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v3

echo "GENERATING MODEL 10/12"
python -u iarpa/runBaseline.py --algorithm resnet50 --train True --num_gpus 4 --num_epochs 5 --batch_size 80 --fine_tunning True --class_weights sklearn_class_weight  --prefix p04 --generator flip --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.resnet50.p01.06.hdf5 --database v3

echo "GENERATING MODEL 11/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 4 --batch_size 28 --class_weights sklearn_class_weight --prefix p07 --generator flip --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v1

echo "GENERATING MODEL 12/12"
python -u iarpa/runBaseline.py --algorithm densenet --train True --num_gpus 4 --num_epochs 5 --batch_size 28 --class_weights sklearn_class_weight --prefix p08 --generator flip --fine_tunning True --load_weights /wdata/working/cnn_checkpoint_weights/weights.v1.densenet.p01.06.hdf5 --database v2

echo "TRAINING FINISHED"
exit
"""

import logging

from fire_risk_classifier.pipeline import Pipeline
from fire_risk_classifier.utils.logger import Logger
from fire_risk_classifier.dataclasses.params import Params
from fire_risk_classifier.utils.arg_parser import ArgumentParser


def main():
    default_params = Params()
    parser = ArgumentParser()

    Logger.initialize_logger()

    pipeline = Pipeline(default_params, parser.get_parser_dict())

    logging.info("Starting testing...")
    pipeline.test_cnn()


if __name__ == "__main__":
    main()

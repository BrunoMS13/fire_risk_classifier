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

import argparse
from typing import Any

from fire_risk_classifier.pipeline import Pipeline
from fire_risk_classifier.dataclasses.params import Params


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Fire Risk Classifier argument parser for training and testing"
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def get_parser_dict(self) -> dict[str, Any]:
        return vars(self.parser.parse_args())


def main():
    default_params = Params()
    parser = ArgumentParser()

    parser.add_argument("--algorithm", default="", type=str)
    parser.add_argument("--train", default="", type=str)
    parser.add_argument("--test", default="", type=str)
    parser.add_argument("--nm", default="", type=str)
    parser.add_argument("--prepare", default="", type=str)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--num_epochs", default=12, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--load_weights", default="", type=str)
    parser.add_argument("--fine_tunning", default="", type=str)
    parser.add_argument("--class_weights", default="", type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--generator", default="", type=str)
    parser.add_argument("--database", default="", type=str)
    parser.add_argument("--path", default="", type=str)

    pipeline = Pipeline(default_params, parser.get_parser_dict())
    params = pipeline.params

    if params.train_cnn:
        pipeline.train_cnn()


if __name__ == "__main__":
    main()

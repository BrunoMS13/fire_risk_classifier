from multiprocessing import cpu_count


class Params:
    def __init__(self):
        self.num_workers = cpu_count()

        self.use_metadata = True

        self.lr_mode = "progressive_drops"

        self.batch_size_cnn = 2

        self.batch_size_lstm = 512
        self.batch_size_eval = 128
        self.metadata_length = 45
        self.num_labels = 5
        self.cnn_last_layer_length = 4096
        self.cnn_lstm_layer_length = 2208

        self.num_gpus = 1

        self.target_img_size = (224, 224)

        self.image_format = "png"

        self.train_cnn = False
        self.generate_cnn_codes = False
        self.train_lstm = False
        self.test_cnn = False
        self.test_lstm = False

        self.algorithm = "resnet"
        self.prefix = ""
        self.model_weights = ""
        self.database = ""
        self.class_weights = "no_weights"
        self.generator = "flip"

        self.fine_tunning = False

        # LEARNING PARAMS
        self.cnn_adam_learning_rate = 1e-4
        self.cnn_adam_loss = "categorical_crossentropy"

        self.cnn_epochs = 12

        self.lstm_adam_learning_rate = 1e-4
        self.lstm_epochs = 100
        self.lstm_loss = "categorical_crossentropy"

        self.directories = {
            "images_directory": "images/ortos2018-IRG-decompressed",
            "annotations_file": "fire_risk_classifier/data/csvs/train.csv",
            "testing_annotations_file": "fire_risk_classifier/data/csvs/test.csv",
            "cnn_checkpoint_weights": "fire_risk_classifier/data/cnn_checkpoint_weights",
        }

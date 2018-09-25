import tensorflow as tf
import json

from data_utils import Data
from models.char_cnn_zhang import CharCNNZhang
from models.char_cnn_kim import CharCNNKim
from models.char_tcn import CharTCN

tf.flags.DEFINE_string("model", "char_cnn_zhang", "Specifies which model to use: kim, tcn or zhang")
tf.flags.DEFINE_string("mode", "train", "Specifies which mode to use: train or test")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

MODEL_DIR = 'data/models'


def train(config, validation_inputs, validation_labels):
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()

    if FLAGS.model == "kim":
        model_key = "char_cnn_kim"
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=config[model_key]["embedding_size"],
                           conv_layers=config[model_key]["conv_layers"],
                           fully_connected_layers=config[model_key]["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=config[model_key]["dropout_p"],
                           optimizer=config[model_key]["optimizer"],
                           loss=config[model_key]["loss"])
    elif FLAGS.model == 'tcn':
        model_key = "char_cnn_kim"
        model = CharTCN(input_size=config["data"]["input_size"],
                        alphabet_size=config["data"]["alphabet_size"],
                        embedding_size=config["char_tcn"]["embedding_size"],
                        conv_layers=config[model_key]["conv_layers"],
                        fully_connected_layers=config[model_key]["fully_connected_layers"],
                        num_of_classes=config["data"]["num_of_classes"],
                        dropout_p=config[model_key]["dropout_p"],
                        optimizer=config[model_key]["optimizer"],
                        loss=config[model_key]["loss"])
    else:
        model_key = "char_cnn_zhang"
        model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=config[model_key]["embedding_size"],
                             conv_layers=config[model_key]["conv_layers"],
                             fully_connected_layers=config[model_key]["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=config[model_key]["threshold"],
                             dropout_p=config[model_key]["dropout_p"],
                             optimizer=config[model_key]["optimizer"],
                             loss=config[model_key]["loss"])

    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])
    model.save(MODEL_DIR)
    print('finish training and saving')


def test(config, testing_inputs, testing_labels):
    if FLAGS.model == "kim":
        model = CharCNNKim(model_dir=MODEL_DIR)
    elif FLAGS.model == 'tcn':
        model = CharTCN(model_dir=MODEL_DIR)
    else:
        model = CharCNNZhang(model_dir=MODEL_DIR)
    result = model.test(testing_inputs=testing_inputs,
               testing_labels=testing_labels,
               batch_size=config["testing"]["batch_size"])
    print('Test result {}'.format(result))
    print('finish loading and testing')


if __name__ == "__main__":
    config = json.load(open("config.json"))

    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    if FLAGS.mode == "train":
        train(config, validation_inputs, validation_labels)
    if FLAGS.mode == "test":
        test(config, validation_inputs, validation_labels)

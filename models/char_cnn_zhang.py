from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout

from models.char_cnn_model import CharCNNModel


class CharCNNZhang(CharCNNModel):
    """
    Class to implement the Character Level Convolutional Neural Network for Text Classification,
    as described in Zhang et al., 2015 (http://arxiv.org/abs/1509.01626)
    """
    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, num_of_classes,
                 threshold, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialization for the Character Level CNN model.

        Args:
            input_size (int): Size of input features
            alphabet_size (int): Size of alphabets to create embeddings for
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            fully_connected_layers (list[list[int]]): List of Fully Connected layers for model
            num_of_classes (int): Number of classes in data
            threshold (float): Threshold for Thresholded ReLU activation function
            dropout_p (float): Dropout Probability
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.threshold = threshold
        self.dropout_p = dropout_p
        super().__init__(optimizer, loss)

    def _get_model(self):
        """
        Build and compile the Character Level CNN model

        Returns: None

        """
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1,
                      self.embedding_size,
                      input_length=self.input_size)(inputs)
        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.threshold)(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = ThresholdedReLU(self.threshold)(x)
            x = Dropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)
        print("CharCNNZhang model built: ")
        return Model(inputs=inputs, outputs=predictions)

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout1D
from keras.layers import Dropout
from keras.layers import Add

from models.char_cnn_model import CharCNNModel


class CharTCN(CharCNNModel):
    """
    Class to implement the Character Level Temporal Convolutional Network (TCN)
    as described in Bai et al., 2018 (https://arxiv.org/pdf/1803.01271.pdf)
    """
    def __init__(self, input_size=None, alphabet_size=None, embedding_size=None,
                 conv_layers=None, fully_connected_layers=None, num_of_classes=None,
                 threshold=None, dropout_p=None, model_dir=None,
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
        self.dropout_p = dropout_p
        super().__init__(optimizer, loss, model_dir)

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
        # Residual blocks with 2 Convolution layers each
        d = 1  # Initial dilation factor
        for cl in self.conv_layers:
            res_in = x
            for _ in range(2):
                # NOTE: The paper used padding='causal'
                x = Convolution1D(cl[0], cl[1], padding='same',
                                  dilation_rate=d, activation='linear')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SpatialDropout1D(self.dropout_p)(x)
                d *= 2  # Update dilation factor
            # Residual connection
            res_in = Convolution1D(filters=cl[0], kernel_size=1,
                                   padding='same', activation='linear')(res_in)
            x = Add()([res_in, x])
        x = Flatten()(x)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl)(x)
            x = Activation('relu')(x)
            x = Dropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)
        print("CharTCN model built: ")
        return Model(inputs=inputs, outputs=predictions)

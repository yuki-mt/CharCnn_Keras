from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout

from models.char_cnn_model import CharCNNModel


class CharCNNKim(CharCNNModel):
    """
    Class to implement the Character Level Convolutional Neural Network
    as described in Kim et al., 2015 (https://arxiv.org/abs/1508.06615)

    Their model has been adapted to perform text classification instead of language modelling
    by replacing subsequent recurrent layers with dense layer(s) to perform softmax over classes.
    """
    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers,
                 num_of_classes, dropout_p,
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
        convolution_output = []
        for num_filters, filter_width in self.conv_layers:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
            pool = GlobalMaxPooling1D(
                name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl, activation='selu', kernel_initializer='lecun_normal')(x)
            x = AlphaDropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)

        print("CharCNNKim model built: ")
        return Model(inputs=inputs, outputs=predictions)

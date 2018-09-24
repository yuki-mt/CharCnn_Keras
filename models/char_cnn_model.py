from keras.callbacks import TensorBoard
from abc import ABCMeta, abstractmethod


class CharCNNModel(metaclass=ABCMeta):
    def __init__(self, optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialization for the Character Level CNN model.

        Args:
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
        self._build_model(optimizer, loss)  # builds self.model variable

    @abstractmethod
    def _get_model(self):
        pass

    def _build_model(self, optimizer, loss):
        """
        Build and compile the Character Level CNN model

        Returns: None
        """
        model = self._get_model()
        model.compile(optimizer=optimizer, loss=loss)
        self.model = model
        self.model.summary()

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size, checkpoint_every=100):
        """
        Training function

        Args:
            training_inputs (numpy.ndarray): Training set inputs
            training_labels (numpy.ndarray): Training set labels
            validation_inputs (numpy.ndarray): Validation set inputs
            validation_labels (numpy.ndarray): Validation set labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_every (int): Interval for logging to Tensorboard

        Returns: None

        """
        # Create callbacks
        tensorboard = TensorBoard(log_dir='./logs',
                                  histogram_freq=checkpoint_every,
                                  batch_size=batch_size,
                                  write_graph=True,
                                  write_grads=True,
                                  write_images=True,
                                  embeddings_freq=checkpoint_every,
                                  embeddings_layer_names=None)
        # Start training
        print("Training model: ")
        self.model.fit(training_inputs, training_labels,
                       validation_data=(validation_inputs, validation_labels),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       callbacks=[tensorboard])

    def test(self, testing_inputs, testing_labels, batch_size):
        """
        Testing function

        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size

        Returns: None

        """
        # Evaluate inputs
        self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)
        # self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)

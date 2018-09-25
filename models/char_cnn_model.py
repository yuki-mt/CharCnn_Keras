from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from abc import ABCMeta, abstractmethod
import os
import json


class CharCNNModel(metaclass=ABCMeta):
    def __init__(self, optimizer='adam', loss='categorical_crossentropy', model_dir=None):
        """
        Initialization for the Character Level CNN model.

        Args:
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
        self._build_model(optimizer, loss, model_dir)

    @abstractmethod
    def _get_model(self):
        pass

    def _build_model(self, optimizer, loss, model_dir):
        if model_dir:
            print('start loading model')
            model = self._load(model_dir)
        else:
            print('start building model')
            model = self._get_model()
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
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

        es = EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

        dirname = 'data/models/chkpt'
        filename = 'char_cnn.{epoch:02d}-{val_loss:.2f}.hdf5'
        cp_cb = ModelCheckpoint(filepath=os.path.join(dirname, filename),
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='auto')
        # Start training
        print("Training model: ")
        self.model.fit(training_inputs, training_labels,
                       validation_data=(validation_inputs, validation_labels),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=1,
                       callbacks=[tensorboard, es, cp_cb])
        self.load_best_weight(dirname)

    def load_best_weight(self, dirname):
        from glob import glob
        target = os.path.join(dirname, '*')
        files = [(f, os.path.getmtime(f)) for f in glob(target)]
        if len(files) != 0:
            newest_model = sorted(files, key=lambda files: files[1])[-1]
            self.model.load_weights(newest_model[0])

    def save(self, dirname):
        base_name = os.path.join(dirname, self.__class__.__name__)
        self.model.save_weights(base_name + '_param.hdf5')
        open(base_name + '_model.json', 'w').write(self.model.to_json())

    def _load(self, dirname):
        base_name = os.path.join(dirname, self.__class__.__name__)
        with open(base_name + '_model.json') as model_json_file:
            model = model_from_json(model_json_file.read())
        model.load_weights(base_name + '_param.hdf5')
        return model

    def test(self, testing_inputs, testing_labels, batch_size):
        result = self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)
        return list(zip(self.model.metrics_names, result))

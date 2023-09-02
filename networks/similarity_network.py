from keras.optimizers import Adam

import conf
from utilities import Utilities
import networks.triplet_mining as triplet_mining
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{")
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


# embedding network and training
class SimilarityNetwork(Utilities):

    def __init__(self, train_loader, validation_loader, checkpoint_dir):
        super().__init__()
        # train_generator's job is to randomly, and lazily, load batches from disk
        self.train_generator = train_loader
        self.validation_generator = validation_loader
        self.exemplar_dim = train_loader.exemplar_dim
        self.checkpoint_dir = checkpoint_dir
        self.embedding_size = conf.embedding_size
        self._network = Sequential()
        self.build_model()
        self.compile_model()

    def build_model(self):
        input_shape = np.expand_dims(self.exemplar_dim, 2)
        self._network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',
                                 input_shape=input_shape))
        self._network.add(Dropout(0.2))
        self._network.add(BatchNormalization())
        self._network.add(MaxPool2D(2, 2))
        self._network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
        self._network.add(Dropout(0.2))
        self._network.add(BatchNormalization())
        self._network.add(MaxPool2D(2, 2))
        self._network.add(Flatten())
        self._network.add(Dense(self.embedding_size))
        self._network.add(Dropout(0.2))

    def compile_model(self):
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        try:
            self._network.compile(loss=triplet_mining.batch_all_triplet_loss(), optimizer=opt)
        except RuntimeError:
            self._network.compile(loss=triplet_mining.batch_all_triplet_loss(), optimizer=opt)
        finally:
            self._network.summary()

    def run_model_training(self):
        self._network.fit(self.train_generator, validation_data=self.validation_generator, epochs=conf.num_epochs,
                          steps_per_epoch=self.train_generator.__len__(),
                          validation_steps=self.validation_generator.__len__())

from keras.optimizers import Adam

import conf
from networks.utilities import Utilities
import networks.triplet_mining as triplet_mining
from keras import callbacks
from keras.models import Sequential
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda, Dropout
import logging
import os

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{")
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


# embedding network and training
class SimilarityNetwork(Utilities):

    def __init__(self, train_loader, validation_loader, test_loader, checkpoint_dir):
        super().__init__()
        # train_generator's job is to randomly, and lazily, load batches from disk
        self.train_set = train_loader
        self.validation_set = validation_loader
        self.test_set = test_loader
        self.exemplar_dim = train_loader.exemplar_dim
        self.checkpoint_dir = checkpoint_dir
        self.embedding_size = conf.embedding_size
        self.callbacks = [callbacks.EarlyStopping(monitor='val_batch_triplet_loss', patience=5, mode='min', verbose=1)]
        self._network = Sequential()
        self.build_model()
        self.compile_model()

    def build_model(self):
        # Add a new dimension for 2D convolution
        input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)
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
        # self._network.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1, epsilon=1e-13)))

    def compile_model(self):
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        try:
            self._network.compile(optimizer=opt, loss=triplet_mining.batch_triplet_loss,
                                  metrics=[triplet_mining.batch_triplet_loss], run_eagerly=True)
        except RuntimeError:
            self._network.compile(optimizer=opt, loss=triplet_mining.batch_triplet_loss, metrics=[
                triplet_mining.batch_triplet_loss])
        finally:
            self._network.summary()

    def run_model_training(self):
        self._network.fit(self.train_set, validation_data=self.validation_set, epochs=conf.n_similarity_epochs,
                          steps_per_epoch=self.train_set.__len__(),
                          validation_steps=self.validation_set.__len__(), callbacks=self.callbacks)

    def evaluate(self):
        self._network.evaluate(self.test_set, steps=self.test_set.__len__())

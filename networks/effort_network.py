import conf
import organize_synthetic_data as osd
from utilities import Utilities
from keras.optimizers import Adam
from keras import losses
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
import tensorflow as tf
import datetime
import math
import os

np_config.enable_numpy_behavior()


class EffortNetwork(Utilities):
    def __init__(self):
        self.sample_count = None
        self.model = None
        self._network = Sequential()
        if os.path.isfile(conf.effort_model_file):
            self.model = tf.keras.models.load_model(conf.effort_model_file)
        else:
            super(EffortNetwork, self).__init__()
            data, labels = osd.load_data(rotations=True, velocities=True)
            self.sample_count = data.shape[0]
            self.make_classes_from_labels(labels)
            self.train_test_split(data, self.classes)
            self.build_model(data)
            self.train_model()

    def build_model(self, data, filter_num=150, kernel_size=(3, 3), strides=(1, 1)):
        input_shape = (data.shape[1], data.shape[2], 1)
        output_layer_size = len(np.unique(self.classes))
        self._network.add(Conv2D(filter_num, kernel_size=kernel_size, strides=strides, activation='ReLU',
                                 input_shape=input_shape))
        self._network.add(MaxPool2D((2, 2)))
        self._network.add(Dropout(0.3))
        self._network.add(Flatten())
        self._network.add(Dense(output_layer_size, activation='softmax'))

    def train_model(self):
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        self._network.compile(loss=losses.sparse_categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
        self._network.summary()
        self._network.save(conf.synthetic_model_file)
        self._network.fit(self.train_ds, epochs=conf.n_epochs, steps_per_epoch=math.ceil(int(self.sample_count *
                          self.train_split)/conf.batch_size), validation_data=self.test_ds, callbacks=[
                          self.tensorboard_callback, self.csv_logger], validation_steps=100)
        self.model = self._network

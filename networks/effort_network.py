import conf
import organize_synthetic_data as osd
from utilities import Utilities
import keras
from keras.optimizers import Adam
from keras import losses
from keras.layers import Input, Conv1D, MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2DTranspose
from tensorflow.python.ops.numpy_ops import np_config
from glob import glob
import numpy as np
import tensorflow as tf
import datetime
import math
import os
np_config.enable_numpy_behavior()


class EffortNetwork(Utilities):
    def __init__(self, two_d_conv=False, model_num=1):
        self.model = None
        self.num_shards = None
        self.STEPS_PER_EPOCH = None
        self._network = Sequential()
        if os.path.isfile(conf.effort_model_file):
            self.model = tf.keras.models.load_model(conf.effort_model_file)
        else:
            super(EffortNetwork, self).__init__()
            # self.data, self.labels = data, labels

            #####
            # x_train = self.data[0:train_size]
            # self.STEPS_PER_EPOCH = len(x_train) // conf.batch_size
            #####
            loss = 'mse'
            if two_d_conv:
                self.data = np.expand_dims(self.data, 3)
                if model_num ==2:
                    conv_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='ReLU',
                                 input_shape=(self.data.shape[1], self.data.shape[2], 1))
                    pool_1 = MaxPool2D(2, 2)
                    self.build_model_2(conv_1, pool_1)
                elif model_num ==3:
                    self.build_model_3(self.data)
                elif model_num ==4:
                    self.build_model_4(self.data)
            elif not two_d_conv:
                # train_ds, test_ds = self.partition_dataset(self.data, self.labels)
                if model_num == 1:
                    conv_1 = Conv1D(filters=120, kernel_size=15, activation='relu', input_shape=(
                        40, 91))
                    pool_1 = MaxPooling1D(pool_size=2)
                    self.build_model_1(conv_1, pool_1)
            self.compile_and_train_model(loss)

    def compile_and_train_model(self, loss):
        try:
            opt = Adam(learning_rate=0.0001, beta_1=0.5)
            self._network.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
            # backup_callback = keras.callbacks.BackupAndRestore(
            #     backup_dir='/tmp/backup')
            # self._network.fit(train_ds, epochs=1, steps_per_epoch=self.STEPS_PER_EPOCH,
            #                       validation_data=test_ds, callbacks=[self.tensorboard_callback, self.csv_logger,
            #                                                           backup_callback])
            # self._network.save(conf.effort_model_file)
            self.model = self._network
        except RuntimeError:
            self.compile_and_train_model(loss)

    def build_model(self, data, filter_num=150, kernel_size=(3, 3), strides=(1, 1)):
        input_shape = (data.shape[1], data.shape[2], 1)
        output_layer_size = len(np.unique(self.classes))
        self._network.add(Conv2D(filter_num, kernel_size=kernel_size, strides=strides, activation='ReLU',
                                 input_shape=input_shape))
        self._network.add(MaxPool2D((2, 2)))
        self._network.add(Dropout(0.3))
        self._network.add(Flatten())
        self._network.add(Dense(output_layer_size, activation='softmax'))

    # avg 560 sec, 50 epoch, val_accuracy: 0.6668
    ###1D cnn: 2s, .9574
    def build_model_1(self, conv_1, pool_1):
        output_layer_size = 4
        self._network.add(conv_1)
        self._network.add(pool_1)
        self._network.add(BatchNormalization())
        self._network.add(Dropout(0.3))
        # self._network.add(Conv1D(filters=4, kernel_size=2, activation='relu'))
        self._network.add(Flatten())
        self._network.add(Dense(output_layer_size, activation='tanh'))

    # avg 220 sec, 50 epoch, val_accuracy: 0.7256
    def build_model_2(self, data, filter_num=150, kernel_size=(3, 3), strides=(1, 1)):
        input_shape = (data.shape[1], data.shape[2], 1)
        output_layer_size = self.labels.shape[1]
        self._network.add(Conv2D(filter_num, kernel_size=kernel_size, strides=strides, activation='ReLU',
                                 input_shape=input_shape))
        self._network.add(BatchNormalization())
        self._network.add(MaxPool2D(2, 2))
        # self._network.add(Conv2D(filter_num/2, kernel_size=kernel_size, strides=strides, activation='ReLU'))
        self._network.add(BatchNormalization())
        self._network.add(Dropout(0.3))
        self._network.add(Flatten())
        self._network.add(Dense(output_layer_size, activation='tanh'))

    def build_model_3(self, data, filter_num=80, kernel_size=(3, 3), strides=(1, 1)):
        input_shape = (data.shape[1], data.shape[2], 1)
        output_layer_size = self.labels.shape[1]
        self._network.add(Conv2D(filter_num, kernel_size=kernel_size, strides=strides, activation='ReLU',
                                 input_shape=input_shape))
        self._network.add(BatchNormalization())
        self._network.add(MaxPool2D(2, 2))
        self._network.add(Conv2D(filter_num/2, kernel_size=kernel_size, strides=strides, activation='ReLU'))
        self._network.add(BatchNormalization())
        # self._network.add(Conv2D(filter_num /4, kernel_size=kernel_size, strides=strides, activation='ReLU'))
        # self._network.add(Conv2D(1, kernel_size=kernel_size, strides=strides, activation='ReLU'))
        self._network.add(Dropout(0.3))
        self._network.add(Flatten())
        self._network.add(Dense(output_layer_size, activation='tanh'))

    def build_model_4(self, data, filter_num=150, kernel_size=(3, 3), strides=(1, 1)):
        input_shape = (data.shape[1], data.shape[2], 1)
        output_layer_size = len(np.unique(self.classes))
        self._network.add(Conv2D(filter_num, kernel_size=kernel_size, strides=strides, activation='ReLU',
                                 input_shape=input_shape))
        self._network.add(MaxPool2D((2, 2)))
        self._network.add(Dropout(0.3))
        self._network.add(Flatten())
        self._network.add(Dense(output_layer_size, activation='softmax'))

import datetime
import math
from collections import Counter

from keras.utils import to_categorical

import conf
import numpy as np
from keras.optimizers import Adam
from keras import losses
from keras.layers import Input, MaxPool1D
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Conv1DTranspose
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers import UpSampling1D
from keras.layers import Concatenate
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import organize_synthetic_data as osd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from math import sqrt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Classifies personality or LMA efforts
def build_and_run_autoencoder(x, y):
    tf.compat.v1.enable_eager_execution()

    # 183 features with velocities, 87 features without velocities
    feature_size = x.shape[2]
    print(f'x_dim: {x.shape}, y_dim: {y.shape}')

    x = np.expand_dims(x, 3)
    train_split = (int)(x.shape[0] * 0.8)
    x_train = x[0:train_split, :, :,:]
    print(f'train size: {x_train.shape}')
    y_train = y[0:train_split,:]

    x_test = x[train_split+1:, :, :]
    print(f'test size: {x_test.shape}')
    y_test = y[train_split+1:,:]

    p_count = y_train.shape[1]

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()


    n_batches = int(x.shape[0] / conf.batch_size_efforts_network)
    train_mode = True
    size_input = (conf.time_series_size, feature_size)  # Shape of an individual input
    if train_mode:
        lma_model = Sequential(
            [Input(shape=size_input, name='input_layer'),
             Dropout(0.5),
             ### ENCODER
             # Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_1'),
             Conv1D(256, kernel_size=15, strides=2, padding='same', activation='ReLU', name='conv_1'),
             Conv1D(64, kernel_size=15, strides=2, padding='same', activation='ReLU', name='conv_2'),
             BatchNormalization(),
             Conv1D(32, kernel_size=15, padding='same', activation='ReLU', name='conv_3'),
             # Conv1D(1, kernel_size=15, padding='same', activation='ReLU', name='conv_4'),
             BatchNormalization(),
             ### DECODER
             Conv1DTranspose(32, kernel_size=15, strides=2, name='conv_decode_1'),
             Conv1DTranspose(64, kernel_size=15, strides=2, name='conv_decode_2'),
             Conv1DTranspose(256, kernel_size=15, strides=2, name='conv_decode_3'),
             Flatten(name='flat_layer'),
             Dense(p_count, activation='tanh', name='output_layer')])
        lma_model.summary()
        # using a small learning rate provides better accuracy
        # B = 0.5 gives better accuracy than 0.9
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        loss = 'mse'

        lma_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # validation_data - tuple on which to evaluate the loss and any model metrics at end of each epoch
        # val_loss correesponds to the value of the cost function for this cross-validation data
        # steps_per_epoch is usually: ceil(num_samples / batch_size)
        # accidental use of steps_per_epoch = 1052, with 1D Convolution,  was leading to val_accuracy of 100% for a couple of epochs earlier in the training sequence
        lma_model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=math.ceil(x_train.shape[0] / conf.batch_size_efforts_network), validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)
        # model.save(conf.synthetic_model_file)
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        # y_pred = onehot_encoder.inverse_transform(y_pred_enc)
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(rotations=False, velocities=True)
    build_and_run_autoencoder(data, labels)
import datetime
from collections import Counter

from keras.utils import to_categorical

import conf
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import UpSampling1D
from keras.layers import Concatenate
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras import losses
import organize_synthetic_data as osd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import math
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def make_classes_from_labels(labels):
    labels_classes = np.empty((labels.shape[0], 1))
    print(f"y shape: {labels_classes.shape}")
    labels_counts = Counter(str(elem) for elem in labels)
    print(f"unique y outputs #: {len(labels_counts)}")
    print(f"unique y outputs distr: {labels_counts}")
    labels_map = {key: value for (value, key) in enumerate(set(tuple(x) for x in labels.tolist()))}
    labels_map_floats = {}
    for key in labels_map.keys():
        # print(f"key: {key}, value: {labels_map[key]}")
        new_key = tuple([float(i) for i in key])
        labels_map_floats[new_key] = labels_map[key]
    # print(f"labels_map: {labels_map_floats}")
    for idx, label in enumerate(labels.tolist()):
        labels_classes[idx] = labels_map_floats[tuple(label)]
    return labels_classes

# Classifies personality or LMA efforts
def build_and_run_autoencoder(x, y):
    onehot_encoder = OneHotEncoder(sparse=False)
    tf.compat.v1.enable_eager_execution()

    # 183 features with velocities, 87 features without velocities
    feature_size = x.shape[2]
    print(f'x_dim: {x.shape}, y_dim: {y.shape}')

    train_split = (int)(x.shape[0] * 0.8)
    x_train = x[0:train_split, :, :]
    print(f'train size: {x_train.shape}')
    y_train = y[0:train_split,:]

    x_test = x[train_split+1:, :, :]
    print(f'test size: {x_test.shape}')
    y_test = y[train_split+1:,:]

    p_count = len(np.unique(y_train))
    y_train = to_categorical(y_train, p_count)
    y_test = to_categorical(y_test, p_count)

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()


    n_batches = int(x.shape[0] / conf.batch_size_efforts_network)
    cnn1d = tf.keras.layers.Conv1D(filters=256, kernel_size=15, padding='same', activation='relu', input_shape=(conf.time_series_size, feature_size))(x_train)
    print(f"cnn1d.shape: {cnn1d.shape}")
    train_mode = True
    size_input = (conf.time_series_size, feature_size, 1)
    if train_mode:
        model = tf.keras.models.Sequential()

        # using a small learning rate provides better accuracy
        # B = 0.5 gives better accuracy than 0.9
        opt = Adam(learning_rate=0.0001, beta_1=0.5)

        # input for a CNN-based neural network: (Batch_size, Spatial_dimensions, feature_maps/channels)
        # we have (conf.batch_size, 150, 87) meaning spatial_dim is 1, so we use a 1D convolution
        model.add(Conv1D(filters=256, kernel_size=15, activation='relu', input_shape=(conf.time_series_size, feature_size)))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="Same", activation='relu', input_shape=size_input))
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="Same", activation='relu', input_shape=size_input))
        # model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(p_count, activation='softmax'))
        # model.add(Dense(p_count, activation='tanh'))

        # model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        model.compile(loss=losses.categorical_crossentropy, optimizer=opt, metrics='accuracy')

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # validation_data - tuple on which to evaluate the loss and any model metrics at end of each epoch
        # val_loss corresponds to the value of the cost function for the test data
        # model.fit(x_train, y_train, batch_size = conf.batch_size, epochs=conf.n_epochs, steps_per_epoch=18, validation_data=(x_test, y_test), callbacks=[tensorboard_callback], validation_steps=100)
        # steps_per_epoch is usually: ceil(num_samples / batch_size)
        print(f"check: {math.ceil(x_train.shape[0] / conf.batch_size_efforts_network)}")
        model.fit(train_data, epochs=conf.n_effort_epochs, steps_per_epoch=math.ceil(x_train.shape[0] / conf.batch_size_efforts_network), validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)
        # model.save(conf.synthetic_model_file)
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        # y_pred = onehot_encoder.inverse_transform(y_pred_enc)
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(rotations=False, velocities=True)
    labels_classes = make_classes_from_labels(labels)

    print(f"x size: {data.shape}, y size: {labels_classes.shape}")
    build_and_run_autoencoder(data, labels_classes)
import datetime

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
from keras.layers import Conv2D
from keras.layers import Conv1DTranspose
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling1D
from keras.layers import Concatenate
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import organize_synthetic_data as osd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.ops.numpy_ops import np_config
import math
np_config.enable_numpy_behavior()


# Classifies personality or LMA efforts
def predict_efforts_cnn(x, y):
    onehot_encoder = OneHotEncoder(sparse=False)
    tf.compat.v1.enable_eager_execution()

    # 183 features
    feature_size = x.shape[2]
    #for window size = 150:
    # x_dim: (692, 150, 87), y_dim: (692, 4)
    print(f'x_dim: {x.shape}, y_dim: {y.shape}')


    # 553 train exemplars each of a 150 x 87 array
    train_split = (int)(x.shape[0] * 0.8)
    x_train = x[0:train_split, :, :]
    print(f'train size: {x_train.shape}')
    y_train = y[0:train_split,:]

    # 138 test_exemplars
    x_test = x[train_split+1:, :, :]
    print(f'test size: {x_test.shape}')
    y_test = y[train_split+1:,:]


    p_count = y_train.shape[1]

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(conf.batch_size).repeat()


    n_batches = int(x.shape[0] / conf.batch_size)
    size_input_2D = (conf.time_series_size, feature_size, 1)
    train_mode = True
    if train_mode:
        model = tf.keras.models.Sequential()

        # using a small learning rate provides better accuracy
        # B = 0.5 gives better accuracy than 0.9
        opt = Adam(learning_rate=0.0001, beta_1=0.5)

        # input for a CNN-based neural network: (Batch_size, Spatial_dimensions, feature_maps/channels)
        # we have (conf.batch_size, 150, 87) meaning spatial_dim is 1, so we use a 1D convolution
        # Out featuremaps = 256
        model.add(Conv1D(filters=256, kernel_size=15, activation='relu', input_shape=(conf.time_series_size, feature_size)))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="Same", activation='relu', input_shape=size_input_2D))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(p_count, activation='tanh'))
        model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # validation_data - tuple on which to evaluate the loss and any model metrics at end of each epoch
        # val_loss correesponds to the value of the cost function for this cross-validation data
        # steps_per_epoch is usually: ceil(num_samples / batch_size)
        model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=math.ceil(x_train.shape[0] / conf.batch_size), validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)

        # model.save(conf.synthetic_model_file)
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        # y_pred = onehot_encoder.inverse_transform(y_pred_enc)
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(rotations=False, velocities=True)
    predict_efforts_cnn(data, labels)
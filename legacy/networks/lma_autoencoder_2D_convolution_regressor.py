import datetime
import math
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
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def partition_dataset(x, y, dataset_sample_count, train_split=0.9, seed=11):
    print(f"dataset shape: {x.shape}")
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=dataset_sample_count, seed=seed)
    train_size = int(train_split * data.shape[0])
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    train_data = train_ds.shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()
    test_data = test_ds.shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()
    print(f"len train: {len(list(train_data))}, len test: {len(list(test_data))}")
    print(tf.data.experimental.cardinality(train_ds))
    return train_data, test_data

# Classifies personality or LMA efforts
def build_and_run_autoencoder(x, y):
    tf.compat.v1.enable_eager_execution()

    # 183 features with velocities, 87 features without velocities
    # add extra dimension so 2D convolutions can be applied to a given exemplar (which will have three dimensions now)
    x = np.expand_dims(x, 3)
    train_data, test_data = partition_dataset(x, y, x.shape[0])
    output_layer_node_count = y.shape[1]
    n_batches = int(x.shape[0] / conf.batch_size_efforts_network)
    train_mode = True
    size_input = (x.shape[1], x.shape[2], x.shape[3])
    # size_input = (conf.time_series_size, feature_size, 1)  # Shape of an individual input
    if train_mode:
        lma_model = Sequential(
            [Input(shape=size_input, name='input_layer'),
             Dropout(0.3),
             ### ENCODER
             Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='ReLU', padding='same', name='conv_1'),
             Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='ReLU', padding='same', name='conv_2'),
             BatchNormalization(),
             Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', padding='same', name='conv_3'),
             Conv2D(1, kernel_size=(1, 1), activation='ReLU', name='conv_4'),
             BatchNormalization(),
             ### DECODER
             Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), name='conv_decode_1'),
             Conv2DTranspose(32, kernel_size=(3, 3), strides=(1, 1), name='conv_decode_2'),
             Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), name='conv_decode_3'),
             Conv2DTranspose(1, kernel_size=(3, 3), name='conv_decode_4'),
             Flatten(name='flat_layer'),
             Dense(output_layer_node_count, activation='tanh', name='output_layer')])
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
        lma_model.fit(train_data, epochs=conf.n_effort_epochs, validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)
        # model.save(conf.synthetic_model_file)
    else:
        train_split = (int)(x.shape[0] * 0.9)
        x_train = x[0:train_split, :, :, :]
        print(f'train size: {x_train.shape}')
        y_train = y[0:train_split, :]

        x_test = x[train_split + 1:, :, :]
        print(f'test size: {x_test.shape}')
        y_test = y[train_split + 1:, :]

        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(rotations=True, velocities=False)
    build_and_run_autoencoder(data, labels)
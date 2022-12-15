import datetime
import math
from collections import Counter

from keras.utils import to_categorical

import conf
import numpy as np
from keras.optimizers import Adam
from keras import losses
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
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers import UpSampling1D
from keras.layers import Concatenate
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import organize_synthetic_data as osd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from math import sqrt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def partition_dataset(x, y, dataset_sample_count, train_split=0.8, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=dataset_sample_count, seed=seed)
    train_size = int(train_split * data.shape[0])
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    y_train_list = [elem[1] for elem in train_ds.as_numpy_iterator()]
    y_test_list = [elem[1] for elem in test_ds.as_numpy_iterator()]
    print(f"train classes #: {len(np.unique(y_train_list))}")
    print(f"test classes #: {len(np.unique(y_test_list))}")
    train_data = train_ds.shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
    test_data = test_ds.shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
    return train_data, test_data

def make_classes_from_labels(labels):
    labels_classes = np.empty((labels.shape[0], 1))
    print(f"y shape: {labels_classes.shape}")
    labels_counts = Counter(str(elem) for elem in labels)
    # print(f"unique y outputs #: {len(labels_counts)}")
    # print(f"unique y outputs distr: {labels_counts}")
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

def partition_dataset_proportionally(x, y):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    for i, (train_index, test_index) in enumerate(sss.split(x, y)):
        print(f"Fold {i}:")
        labels_counts = Counter(str(elem) for elem in y[train_index])
        labels_counts_2 = Counter(str(elem) for elem in y[test_index])
        print(f"unique y outputs #: {len(labels_counts)}")
        print(f"unique y outputs 2: {len(labels_counts_2)}")
        x_train = x[train_index, :, :, :]
        y_train = y[train_index, :]
        x_test = x[test_index, :, :]
        y_test = y[test_index, :]
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(
            conf.batch_size).repeat()
        test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(
            conf.batch_size).repeat()
    return train_data, test_data

# Classifies personality or LMA efforts
def predict_efforts_cnn(x, y):
    onehot_encoder = OneHotEncoder(sparse=False)
    tf.compat.v1.enable_eager_execution()
    y = make_classes_from_labels(y)

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

    output_layer_node_count = len(np.unique(y))
    train_data, test_data = partition_dataset(x, y, x.shape[0])
    train_mode = True
    size_input = (x.shape[1], x.shape[2], x.shape[3]) # Shape of one sample: (conf.time_series_size, feature_size, 1)
    if train_mode:
        lma_model = Sequential(
            [Input(shape=size_input, name='input_layer'),
             # Use 256 conv. filters of size 3x3 and shift them in 1-pixel steps
             Conv2D(183, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_1'),
             # Max pooling with a window size of 2x2 pixels. Default stride equals window size, i.e., no window overlap
             MaxPool2D((2, 2), name='maxpool_1'),
             # Deactivate random subset of 30% of neurons in the previous layer in each learning step to avoid overfitting
             Dropout(0.3),
             # Reshape the input tensor provided the previous layer into a vector (1-dim. array) required by dense layers
             Flatten(name='flat_layer'),
             # A dense layer of 33 neurons ("dense" implies complete connections to all of its inputs)
             Dense(output_layer_node_count, activation='softmax', name='output_layer')])

        # using a small learning rate provides better accuracy
        # B = 0.5 gives better accuracy than 0.9
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        loss = 'mse'
        loss_2 = losses.sparse_categorical_crossentropy
        lma_model.compile(loss=loss_2, optimizer=opt, metrics=['accuracy'])
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # validation_data - tuple on which to evaluate the loss and any model metrics at end of each epoch
        # val_loss corresponds to the value of the cost function for this cross-validation data
        # steps_per_epoch is usually: ceil(num_samples / batch_size)
        lma_model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=math.ceil(x_train.shape[0] / conf.batch_size), validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)
        # model.save(conf.synthetic_model_file)
        lma_model.summary()
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        # y_pred = onehot_encoder.inverse_transform(y_pred_enc)
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(rotations=True, velocities=False)
    predict_efforts_cnn(data, labels)
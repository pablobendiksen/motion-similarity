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

def partition_dataset(x, y, dataset_sample_count, train_split=0.8, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=dataset_sample_count, seed=seed)
    train_size = int(train_split * x.shape[0])
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    y_train_list = [elem[1] for elem in train_ds.as_numpy_iterator()]
    y_test_list = [elem[1] for elem in test_ds.as_numpy_iterator()]
    print(f"train classes #: {len(np.unique(y_train_list))}")
    print(f"test classes #: {len(np.unique(y_test_list))}")
    train_data = train_ds.shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
    test_data = test_ds.shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
    return train_data, test_data

# Classifies personality or LMA efforts
def predict_efforts_cnn(x, y):
    onehot_encoder = OneHotEncoder(sparse=False)
    tf.compat.v1.enable_eager_execution()
    y=make_classes_from_labels(y)
    print(f'x_dim: {x.shape}, y_dim: {y.shape}')
    # 183 features with velocities, 87 features without velocities
    train_split = (int)(x.shape[0] * 0.8)
    x_train = x[0:train_split, :, :]
    print(f'train size: {x_train.shape}')
    x_test = x[train_split+1:, :, :]
    print(f'test size: {x_test.shape}')
    y_test = y[train_split+1:,:]

    output_layer_node_count = len(np.unique(y))
    train_data, test_data = partition_dataset(x, y, x.shape[0])
    train_mode = True
    size_input = (x.shape[1], x.shape[2])
    if train_mode:
        model = tf.keras.models.Sequential()
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        # input for a CNN-based neural network: (Batch_size, Spatial_dimensions, feature_maps/channels)
        model.add(Conv1D(filters=256, kernel_size=15, activation='relu', input_shape=(size_input)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        # p_count = 4; one per effort
        model.add(Dense(output_layer_node_count, activation='softmax'))
        model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=opt, metrics='accuracy')
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=math.ceil(x_train.shape[0] / conf.batch_size), validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)
        y_pred_enc = model.predict(x_test)[0]
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(rotations=True, velocities=True)
    labels_classes = make_classes_from_labels(labels)

    print(f"x size: {data.shape}, y size: {labels_classes.shape}")
    predict_efforts_cnn(data, labels_classes)
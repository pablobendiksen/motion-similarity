import organize_synthetic_data as osd
import conf
from collections import Counter
from matplotlib import pyplot as plt
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
from keras.callbacks import CSVLogger
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
import tensorflow as tf
import datetime
import math
import time
np_config.enable_numpy_behavior()


class TimeCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = time.process_time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append((epoch, time.process_time() - self.timetaken))

    def on_train_end(self, logs={}):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.show()


def make_classes_from_labels(labels):
    labels_classes = np.empty((labels.shape[0], 1))
    print(f"y shape: {labels_classes.shape}")
    # labels_map={label: class_num}
    labels_map = {label: class_num for (class_num, label) in enumerate(set(tuple(x) for x in labels.tolist()))}
    # get class_num to populate new labels/classes array
    for idx, label in enumerate(labels.tolist()):
        labels_classes[idx] = labels_map[tuple(label)]
    return labels_classes


def _visualize_class_distribution(train_ds, test_ds, train_title, test_title):
    train_test_labels = []
    if type(train_ds).__module__ == np.__name__:
        if train_ds.ndim == 2 or test_ds.ndim == 2:
            train_ds = train_ds.flatten()
            test_ds = test_ds.flatten()
        train_test_labels.append(train_ds)
        train_test_labels.append(test_ds)
    else:
        train_test_labels.append(np.concatenate([y for x, y in train_ds], axis=0))
        train_test_labels.append(np.concatenate([y for x, y in test_ds], axis=0))
    for count, labels_array in enumerate(train_test_labels):
        labels_array.sort()
        labels, counts = zip(*Counter(labels_array).items())
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, counts, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.xlabel('class index')
        plt.ylabel('frequency')
        if count == 0:
            plt.title(f"{train_title} with n = {len(labels_array)}")
        else:
            plt.title(f"{test_title} with n = {len(labels_array)}")
        plt.show(block=False)
        plt.pause(3)
        plt.close()

# does not achieve 1.0 val accuracy in 50 epochs even though data is shuffled before splitting
def partition_dataset_sklearn_np(x, y, train_split=0.8, seed=0):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_split, random_state=seed,
                                                        shuffle=True)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(
        conf.batch_size_efforts_network).repeat()
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(
        conf.batch_size_efforts_network).repeat()
    return train_data, test_data


def partition_dataset_tf_ds(x, y, train_split=0.8, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=conf.buffer_size, seed=seed)
    train_size = int(train_split * x.shape[0])
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    _visualize_class_distribution(train_ds, test_ds, "Train Data tf.ds.shuffle", "Test Data tf.ds.shuffle")
    y_train_list = [elem[1] for elem in train_ds.as_numpy_iterator()]
    y_test_list = [elem[1] for elem in test_ds.as_numpy_iterator()]
    print(f"train classes #: {len(np.unique(y_train_list))}")
    print(f"test classes #: {len(np.unique(y_test_list))}")
    train_ds = train_ds.shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()
    test_ds = test_ds.shuffle(conf.buffer_size).batch(conf.batch_size_efforts_network).repeat()
    return train_ds, test_ds

# does not achieve 1.0 val accuracy in 50 epochs
def train_model_stratified_shuffle_split(x, y, train_split=0.8, n_splits=1, seed=0):
    sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_split, random_state=seed)
    for i, (train_index, test_index) in enumerate(sss.split(x, y)):
        print(f"Fold {i}:")
        size_input = (x.shape[1], x.shape[2], x.shape[3])
        output_layer_node_count = len(np.unique(y))
        labels_counts = Counter(str(elem) for elem in y[train_index])
        labels_counts_2 = Counter(str(elem) for elem in y[test_index])
        print(f"unique y outputs train: {len(labels_counts)}")
        print(f"unique y outputs test: {len(labels_counts_2)}")
        x_train = x[train_index, :, :, :]
        y_train = y[train_index, :]
        x_test = x[test_index, :, :]
        y_test = y[test_index, :]
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(
            conf.batch_size_efforts_network).repeat()
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(
            conf.batch_size_efforts_network).repeat()
        # _visualize_class_distribution(train_ds, test_ds, "Train Data tf.ds.shuffle", "Test Data tf.ds.shuffle")
        network_model = build_model(size_input, 150, (3, 3), (1, 1), output_layer_node_count)
        train_model(network_model, train_ds, test_ds, x_train.shape[0], False)


def build_model(input_size, filter_num=150, kernel_size=(3, 3), strides=(1, 1), output_layer_size=33):
    network_model = Sequential(
        [Input(shape=input_size, name='input_layer'),
         Conv2D(filter_num, kernel_size=kernel_size, strides=strides, activation='ReLU', name='conv_1'),
         # Max pooling with a window size of 2x2 pixels. Default stride equals window size, i.e., no window overlap
         MaxPool2D((2, 2), name='maxpool_1'),
         # Deactivate random subset of neurons in the previous layer in each learning step to avoid overfitting
         Dropout(0.3),
         Flatten(name='flat_layer'),
         Dense(output_layer_size, activation='softmax', name='output_layer')])
    return network_model


def train_model(network_model, train_data, test_data, exemplar_count, fold_no=False):
    lma_model = network_model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    loss_2 = losses.sparse_categorical_crossentropy
    lma_model.compile(loss=loss_2, optimizer=opt, metrics=['accuracy'])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    timetaken = TimeCallBack()
    lma_model.summary()
    # csv_logger = CSVLogger('log.csv', append=True, separator=';')
    # lma_model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=math.ceil(exemplar_count / conf.batch_size),
    #               validation_data=test_data, callbacks=[csv_logger], validation_steps=100)
    lma_model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=math.ceil(exemplar_count / conf.batch_size_efforts_network),
                  validation_data=test_data, callbacks=[timetaken], validation_steps=100)
    # model.save(conf.synthetic_model_file)


# Classify LMA efforts
def predict_efforts_cnn(x, y):
    tf.compat.v1.enable_eager_execution()
    print(f"label ex: {y[1]}")
    y = make_classes_from_labels(y)
    print(f"class ex: {y[1]}")
    # 183 features with velocities, 87 features without velocities
    x = np.expand_dims(x, 3)
    train_split = (int)(x.shape[0] * 0.8)
    x_train = x[0:train_split, :, :, :]
    x_test = x[train_split + 1:, :, :]
    output_layer_node_count = len(np.unique(y))
    train_data, test_data = partition_dataset_tf_ds(x, y)
    train_mode = True
    # Shape of one sample: (conf.time_series_size, feature_size, 1)
    size_input = (x.shape[1], x.shape[2], x.shape[3])
    if train_mode:
        network_model = build_model(size_input, 150, (3, 3), (1, 1), output_layer_node_count)
        train_model(network_model, train_data, test_data, x_train.shape[0], False)
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        print(y_pred_enc)


if __name__ == "__main__":
    data, labels = osd.load_data(rotations=True, velocities=True)
    predict_efforts_cnn(data, labels)

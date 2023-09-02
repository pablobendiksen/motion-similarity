import conf
from networks.utilities import Utilities
from keras.optimizers import Adam
from keras.layers import Input, Conv1D, MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras import models
from keras import callbacks
import numpy as np
import tensorflow as tf
import logging
import csv
import os

# np_config.enable_numpy_behavior()

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{")
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


class EffortNetwork(Utilities):
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    def __init__(self, train_generator, validation_generator, test_generator, checkpoint_dir, two_d_conv=False):
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        self.checkpoint_dir = checkpoint_dir
        self.exemplar_dim = train_generator.exemplar_dim
        self.output_layer_size = self.train_generator.labels.shape[1]
        self.model = None
        self.STEPS_PER_EPOCH = None
        self.checkpoint = None
        self.callbacks = [callbacks.EarlyStopping(monitor='val_mse', patience=5, mode='min')]
        self._network = Sequential()
        if os.path.isfile(conf.effort_model_file):
            self.model = tf.keras.models.load_model(conf.effort_model_file)
        else:
            super().__init__()
            loss = 'mse'
            if two_d_conv:
                self.build_model_2d_conv()
            else:
                self.build_model()
            self.compile_model(loss)

    def build_model(self, filters=160, kernel_size=15, strides=1):
        self._network.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu',
                                 input_shape=self.exemplar_dim))
        self._network.add(MaxPooling1D(pool_size=2))
        self._network.add(BatchNormalization())
        self._network.add(Dropout(0.3))
        # self._network.add(Conv1D(filters=4, kernel_size=2, activation='relu'))
        self._network.add(Flatten())
        self._network.add(Dense(self.output_layer_size, activation='tanh'))
        self._network.summary()

    def build_model_2d_conv(self, filters=120, kernel_size=(3, 3), strides=(1, 1)):
        input_shape = np.expand_dims(self.exemplar_dim, 2)
        self._network.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu',
                                 input_shape=input_shape))
        self._network.add(MaxPool2D(2, 2))
        self._network.add(BatchNormalization())
        # self._network.add(Conv2D(filter_num/2, kernel_size=kernel_size, strides=strides, activation='ReLU'))
        self._network.add(Dropout(0.3))
        self._network.add(Flatten())
        self._network.add(Dense(self.output_layer_size, activation='tanh'))

    def compile_model(self, loss):
        try:
            opt = Adam(learning_rate=0.0001, beta_1=0.5)
            self._network.compile(loss=loss, optimizer=opt, metrics=['mse'])
            self.model = self._network
        except RuntimeError:
            self.compile_model(loss)

    def run_model_training(self):
        try:

            history = self.model.fit(self.train_generator, validation_data=self.validation_generator,
                                     validation_steps=self.validation_generator.__len__(), epochs=conf.n_epochs,
                                     workers=4, use_multiprocessing=True,
                                     steps_per_epoch=self.train_generator.__len__(), callbacks=self.callbacks)

            self.model.save(self.checkpoint_dir)
            self.model.save_weights(self.checkpoint_dir)
            return history
        except RuntimeError as run_err:
            logging.error(f"RuntimeError for job {conf.num_task}, attempting training restoration - {run_err} ")
            history = self.model.fit(self.train_generator, validation_data=self.validation_generator,
                                     validation_steps=self.validation_generator.__len__(), epochs=conf.n_epochs,
                                     workers=1, use_multiprocessing=False,
                                     steps_per_epoch=self.train_generator.__len__())
            self.model.save(self.checkpoint_dir)
            self.model.save_weights(self.checkpoint_dir)
            return history

    def write_out_training_results(self, total_time):
        # run test data through trained model
        saved_model = models.load_model(self.checkpoint_dir)
        saved_model.load_weights(self.checkpoint_dir)
        test_loss, metric = saved_model.evaluate(self.test_generator)
        print(f'Test loss: {test_loss}, Metric (MSE): {metric}')
        csv_file = os.path.join(conf.output_metrics_dir, f'{conf.num_task}_{conf.window_delta}.csv')
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                header_row = next(reader)
                if header_row == ['Percent Copied', 'Index', 'Sliding Window Size', 'BVH File Num', 'Exemplar Num',
                                  'Val Loss', 'Metric (MSE)', 'Training Time']:
                    append_header = False
                else:
                    # incorrect header_row
                    append_header = True
        else:
            append_header = True

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if append_header:
                writer.writerow(['Percent Copied', 'Index', 'Sliding Window Size', 'BVH File Num', 'Exemplar Num',
                                 'Val Loss', 'Metric (MSE)', 'Training Time'])
            print(f"Writing out to: {csv_file}")
            writer.writerow([conf.percent_files_copied, conf.task_num, conf.window_delta, conf.bvh_file_num,
                             conf.exemplar_num,
                             test_loss, metric, total_time])

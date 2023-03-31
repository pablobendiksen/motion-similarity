import conf
from utilities import Utilities
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
# from tensorflow.python.ops.numpy_ops import np_config
from keras.callbacks import EarlyStopping
from keras.callbacks import BackupAndRestore
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

# callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='auto')
backup_restore = BackupAndRestore(backup_dir="/tmp/backup")


class EffortNetwork(Utilities):
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # class InterruptingCallback(tf.keras.callbacks.Callback):
    #     def on_epoch_begin(self, epoch, logs=None):
    #         if epoch == 4:
    #             raise RuntimeError('Interrupting!')

    def __init__(self, two_d_conv=False, model_num=1):
        self.model = None
        self.STEPS_PER_EPOCH = None
        self.checkpoint = None
        self._network = Sequential()
        if os.path.isfile(conf.effort_model_file):
            self.model = tf.keras.models.load_model(conf.effort_model_file)
        else:
            super(EffortNetwork, self).__init__()
            loss = 'mse'
            if two_d_conv:
                self.data = np.expand_dims(self.data, 3)
                if model_num == 2:
                    conv_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='ReLU',
                                    input_shape=(100, 91, 1))
                    pool_1 = MaxPool2D(2, 2)
                    self.build_model_2(conv_1, pool_1)
                elif model_num == 3:
                    self.build_model_3(self.data)
                elif model_num == 4:
                    self.build_model_4(self.data)
            elif not two_d_conv:
                if model_num == 1:
                    self.build_model_1()
            self.compile_model(loss)

    def compile_model(self, loss):
        try:
            opt = Adam(learning_rate=0.0001, beta_1=0.5)
            self._network.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
            self.model = self._network
        except RuntimeError:
            self.compile_model(loss)

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
    def build_model_1(self):
        self._network.add(Conv1D(filters=160, kernel_size=15, activation='relu', input_shape=(
            100, 91)))
        self._network.add(MaxPooling1D(pool_size=2))
        self._network.add(BatchNormalization())
        self._network.add(Dropout(0.3))
        # self._network.add(Conv1D(filters=4, kernel_size=2, activation='relu'))
        self._network.add(Flatten())
        self._network.add(Dense(4, activation='tanh'))
        self._network.summary()

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
        self._network.add(Conv2D(filter_num / 2, kernel_size=kernel_size, strides=strides, activation='ReLU'))
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

    def run_model_training(self, effort_network, train_generator, validation_generator, checkpoint_dir):
        try:
            history = effort_network.model.fit(train_generator, validation_data=validation_generator,
                                     validation_steps=validation_generator.__len__(), epochs=conf.n_epochs,
                                     workers=4, use_multiprocessing=True,
                                     steps_per_epoch=train_generator.__len__(), callbacks=[early_stopping])
            effort_network.model.save(checkpoint_dir)
            effort_network.model.save_weights(checkpoint_dir)
            return history
        except RuntimeError as run_err:
            logging.error(f"RuntimeError for job {conf.task_num}, attempting training restoration - {run_err} ")
            history = effort_network.model.fit(train_generator, validation_data=validation_generator,
                                     validation_steps=validation_generator.__len__(), epochs=conf.n_epochs,
                                     workers=1, use_multiprocessing=False,
                                     steps_per_epoch=train_generator.__len__(), callbacks=[
                                                                                           early_stopping])
            effort_network.model.save(checkpoint_dir)
            effort_network.model.save_weights(checkpoint_dir)
            return history

    def write_out_eval_accuracy(self, validation_generator, task_num, checkpoint_dir, total_time):
        # test stored model use
        saved_model = models.load_model(checkpoint_dir)
        saved_model.load_weights(checkpoint_dir)
        test_loss, test_acc = saved_model.evaluate(validation_generator)
        print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
        num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
        csv_file = os.path.join(conf.metrics_dir, f'{conf.task_num}.csv')
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                header_row = next(reader)
                if header_row == ['Percent Copied', 'Index','Sliding Window Size', 'BVH File Num', 'Exemplar Num',
                                  'Val Loss', 'Val Accuracy', 'Training Time']:
                    append_header = False
                else:
                    # incorrect header_row
                    append_header = True
        else:
            append_header = True

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if append_header:
                writer.writerow(['Percent Copied', 'Index','Sliding Window Size', 'BVH File Num', 'Exemplar Num',
                                  'Val Loss', 'Val Accuracy', 'Training Time'])
            writer.writerow([conf.percent_files_copied, task_num, conf.window_delta, conf.bvh_file_num,
                             conf.exemplar_num,
                             test_loss, test_acc, int(total_time)])

        # Write out to csv file
        # with open(os.path.join(conf.metrics_dir, f'{conf.task_num}.csv'), "w", newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['Percent Copied', 'Index', 'Sliding Window Size', 'BVH File Num', 'Exemplar Num',
        #                      'Val Loss',
        #                      'Val Accuracy', 'Training Time'])
        #     writer.writerow([conf.percent_files_copied, task_num, conf.window_delta, conf.bvh_file_num,
        #                      conf.exemplar_num,
        #         test_loss, test_acc, int(total_time)])

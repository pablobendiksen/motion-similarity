import datetime

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
from sklearn.preprocessing import StandardScaler
from math import sqrt
from tensorflow.python.ops.numpy_ops import np_config
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
    x = np.expand_dims(x, 3)
    train_split = (int)(x.shape[0] * 0.8)
    x_train = x[0:train_split, :, :,:]
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
    train_mode = True
    size_input = (150, 183, 1)  # Shape of an individual input
    if train_mode:
        lma_model = Sequential(
            [Input(shape=size_input, name='input_layer'),
             Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_1'),
             # Use 64 conv. filters of size 3x3 and shift them in 1-pixel steps
             MaxPool2D((2, 2), name='maxpool_1'),
             # Max pooling with a window size of 2x2 pixels. Default stride equals window size, i.e., no window overlap
             Dropout(0.3),
             # Deactivate random subset of 30% of neurons in the previous layer in each learning step to avoid overfitting
             Flatten(name='flat_layer'),
             # Reshape the input tensor provided the previous layer into a vector (1-dim. array) required by dense layers
             Dense(100, activation='ReLU', name='dense_layer_1'),
             # A dense layer of 100 neurons ("dense" implies complete connections to all of its inputs)
             Dense(p_count, activation='softmax', name='output_layer')])

        # using a small learning rate provides better accuracy
        # B = 0.5 gives better accuracy than 0.9
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        loss = 'mse'
        loss_2 = losses.sparse_categorical_crossentropy

        lma_model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # validation_data - tuple on which to evaluate the loss and any model metrics at end of each epoch
        # val_loss correesponds to the value of the cost function for this cross-validation data
        # steps_per_epoch is usually: ceil(num_samples / batch_size)
        # accidental use of steps_per_epoch = 1052, with 1D Convolution,  was leading to val_accuracy of 100% for a couple of epochs earlier in the training sequence
        lma_model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=200, validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=100)
        # model.save(conf.synthetic_model_file)
    else:
        model = tf.keras.models.load_model(conf.synthetic_model_file)

        y_pred_enc = model.predict(x_test)[0]
        # y_pred = onehot_encoder.inverse_transform(y_pred_enc)
        print(y_test[0])
        print(y_pred_enc)

if __name__ == "__main__":
    data, labels = osd.load_data(velocities=True)
    predict_efforts_cnn(data, labels)
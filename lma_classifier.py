import conf

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
from keras.layers import UpSampling1D
from keras.layers import Concatenate
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import organize_synthetic_data as osd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from math import sqrt


# Classifies personality or LMA efforts

onehot_encoder = OneHotEncoder(sparse=False)
tf.compat.v1.enable_eager_execution()

x, y = osd.load_data()
feature_size = x.shape[2]


# train the standardization
# for i in range(len(x)):
#     scaler = StandardScaler()
#     scaler = scaler.fit(x[i])
#     # print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
#     # standardization the dataset and print the first 5 rows
#     normalized_x = scaler.transform(x[i])
#     x[i] =normalized_x



# # train the standardization
# scaler = StandardScaler()
# scaler = scaler.fit(y)
# print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
# # standardization the dataset and print the first 5 rows
# normalized_y = scaler.transform(y)
#
# x = normalized_x
# y = normalized_y



#
train_split = (int)(x.shape[0] * 0.8)
x_train = x[0:train_split, :]
y_train = y[0:train_split,:]

x_test = x[train_split+1:, :]
y_test = y[train_split+1:,:]




p_count = y_train.shape[1]


train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(conf.batch_size).repeat()


n_batches = int(x.shape[0] / conf.batch_size)

train_mode = True
if train_mode:
    model = tf.keras.models.Sequential()

    # using a small learning rate provides better accuracy
    # B = 0.5 gives better accuracy than 0.9
    opt = Adam(lr=0.0001, beta_1=0.5)

    model.add(Conv1D(filters=256, kernel_size=15, activation='relu', input_shape=(conf.time_series_size, feature_size)))
    # model.add(Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=(conf.time_series_size, feature_size)))

    # model.add(Conv1D(filters=64, kernel_size=15))
    # model.add(Conv1D(filters=64, kernel_size=15))


    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # out(ts , feature_size)
    out = model.add(Dense(p_count, activation='tanh'))

    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.fit(train_data, epochs=conf.n_epochs, steps_per_epoch=200, validation_data=test_data, validation_steps=50)

    model.save(conf.synthetic_model_file)
else:
    model = tf.keras.models.load_model(conf.synthetic_model_file)

    y_pred_enc = model.predict(x_test)[0]
    # y_pred = onehot_encoder.inverse_transform(y_pred_enc)
    print(y_pred_enc)
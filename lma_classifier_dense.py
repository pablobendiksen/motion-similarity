from tensorflow import keras
from keras.layers import Dense, Flatten
import tensorflow as tf
import numpy as np
import conf
import datetime

# load mocap preprocessed data for training and testing
x, y = np.load('data/organized_synthetic_data_velocities_150.npy'), np.load('data/organized_synthetic_labels_150.npy')

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

num_epochs = 100
eta = 0.1
decay_factor = 0.98
feature_size = x.shape[2]
size_input = (conf.time_series_size, feature_size)
size_hidden_1 = 200
size_hidden_2 = 200
size_hidden_3 = 200
size_output = y_train.shape[1]
print(f"size_output: {size_output}")

num_train_exemplars = x_train.shape[0]
print(f"num train exemplars: {num_train_exemplars}")

# Build the model (computational graph)
input_layer = keras.Input(shape=size_input, name="input_layer")
flat_input_layer = Flatten(input_shape=size_input, name='flat_input_layer')(input_layer)
hidden_layer_1 = Dense(size_hidden_1, activation='ReLU', name='hidden_layer_1')(flat_input_layer)
hidden_layer_2 = Dense(size_hidden_2, activation='ReLU', name='hidden_layer_2')(hidden_layer_1)
hidden_layer_3 = Dense(size_hidden_2, activation='ReLU', name='hidden_layer_3')(hidden_layer_2)
output_layer = Dense(size_output, activation='softmax', name='output_layer')(hidden_layer_3)

lma_model = keras.Model(inputs=input_layer, outputs=output_layer)

# Print a summary of the model's layers
lma_model.summary()

learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=eta, decay_steps=num_train_exemplars, decay_rate=decay_factor, staircase=True)
optimizer_adam = keras.optimizers.Adam(learning_rate=learning_rate_schedule)
optimizer_sgd = keras.optimizers.SGD(learning_rate=learning_rate_schedule)
lma_model.compile(loss='mse', optimizer=optimizer_adam, metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(conf.buffer_size).batch(conf.batch_size).repeat()
# validation_data - tuple on which to evaluate the loss and any model metrics at end of each epoch
# val_loss correesponds to the value of the cost function for this cross-validation data
lma_model.fit(train_data, epochs=num_epochs, steps_per_epoch=200, validation_data=test_data, callbacks=[tensorboard_callback], validation_steps=50)
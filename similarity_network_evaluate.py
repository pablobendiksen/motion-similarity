from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.layers import Conv2D
from keras.src.layers import BatchNormalization
from keras.src.layers import Dense
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers import Dropout
from keras.src.layers import Flatten
from keras.src.optimizers import Adam
from keras.src.models import Sequential

import networks.custom_losses as custom_losses
from networks.utilities import Utilities
from keras import callbacks
# tf.config.experimental_run_functions_eagerly(True)
import logging
import os
import time
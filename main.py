from datetime import datetime
import keras
from keras.layers import Flatten, Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout
from keras.optimizers import Adam

from networks.effort_network import EffortNetwork
from keras.callbacks import EarlyStopping
from keras.callbacks import BackupAndRestore
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import models, Sequential
from networks.generator import MotionDataGenerator
import organize_synthetic_data as osd
import conf
import os

#generator params
params = {'batch_dim': (100, 91),
          'batch_size': conf.batch_size,
          'shuffle': True}

if __name__ == '__main__':
    index = 1
    checkpoint_dir = f'model_checkpoint_{index}'
    batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
    print(f"number of batches: {len(labels_dict.keys())}")
    print(f"type: {type(labels_dict[1])}")
    train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict, **params)
    print(os.cpu_count())
    validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict, **params)
    effort_network = EffortNetwork(two_d_conv=False, model_num=1)
    effort_network.run_model_training(effort_network, train_generator, validation_generator, index,
                                            checkpoint_dir)
    effort_network.write_out_eval_accuracy(validation_generator, index, checkpoint_dir)

    saved_model = models.load_model(checkpoint_dir)
    saved_model.load_weights(checkpoint_dir)
    test_loss, test_acc = saved_model.evaluate(validation_generator)
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')


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
import logging
import os
logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.basename(__file__) + '.log',
                        format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                        style="{")


def run_model_training(effort_network, train_generator, validation_generator):
    try:
        effort_network.model.fit(train_generator, validation_data=validation_generator,
                                 validation_steps=validation_generator.__len__(), epochs=conf.n_epochs,
                                 steps_per_epoch=train_generator.__len__(), callbacks=[backup_restore,
                                                                                       early_stopping])
        effort_network.model.save('model_checkpoint')
        effort_network.model.save_weights('model_checkpoint')
    except RuntimeError as run_err:
        logging.error(f"RuntimeError, attempting training restoration - {run_err} ")
        run_model_training(effort_network, train_generator, validation_generator)

#callbacks
# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='auto')
backup_restore = BackupAndRestore(backup_dir="/tmp/backup")

#generator params
params = {'batch_dim': (100, 91),
          'batch_size': conf.batch_size,
          'shuffle': True}

if __name__ == '__main__':
    partition, labels_dict = osd.load_data(rotations=True, velocities=False)
    print(f"number of exemplars: {len(labels_dict.keys())}")
    train_generator = MotionDataGenerator(partition['train'], labels_dict, **params)
    validation_generator = MotionDataGenerator(partition['validation'], labels_dict, **params)
    effort_network = EffortNetwork(two_d_conv=False, model_num=1)
    run_model_training(effort_network, train_generator, validation_generator)
    #test stored model use
    saved_model = models.load_model('model_checkpoint')
    saved_model.load_weights('model_checkpoint')
    test_loss, test_acc = saved_model.evaluate(validation_generator)
    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

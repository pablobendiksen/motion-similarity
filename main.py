import random
from networks.effort_network import EffortNetwork
from keras.callbacks import EarlyStopping
from networks.generator import MotionDataGenerator
import organize_synthetic_data as osd
import conf

params = {'batch_dim': (40, 91),
          'batch_size': conf.batch_size,
          'shuffle': True}

if __name__ == '__main__':
    partition, labels_dict = osd.load_data(rotations=True, velocities=False)
    print(f"number of exemplars: {len(labels_dict.keys())}")
    train_generator = MotionDataGenerator(partition['train'], labels_dict, **params)
    validation_generator = MotionDataGenerator(partition['validation'], labels_dict, **params)
    effort_network = EffortNetwork(two_d_conv=False, model_num=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='auto')
    effort_network.model.fit(train_generator, validation_data=validation_generator,
                             validation_steps=validation_generator.__len__(), epochs=conf.n_epochs,
                             steps_per_epoch=train_generator.__len__(), callbacks=[early_stopping])


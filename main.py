import random
from networks.effort_network import EffortNetwork
from keras.callbacks import EarlyStopping
from networks.generator import MotionDataGenerator
import organize_synthetic_data as osd
import conf
import numpy as np
from glob import glob
import os

params = {'dim': (40, 91),
          'batch_size': conf.batch_size,
          'shuffle': True}


def generator(list_idxs, labels, batch_size=conf.batch_size):
    batch_features = np.zeros((batch_size, 40, 91))
    batch_labels = np.zeros((batch_size, 4))
    print(len(list_idxs))
    while True:
        for i in range(batch_size):
            # choose random index in features
            idx = random.choice(list_idxs)
            data_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/data_tmp"
            path = glob(os.path.join(data_dir, f'*_{idx}.npy'))
            batch_features[i] = np.load(path[0])
            batch_labels[i] = labels[idx]
        yield batch_features, batch_labels


if __name__ == '__main__':
    # data has shape (160454, 100, 87) corresponding to (num_exemplars, num_lines_per_exemplar, num_features)
    # labels have shape corresponding to (num_exemplars, num_label_components)
    partition, labels_dict = osd.load_data(rotations=True, velocities=False)
    print(partition)
    print(f"labels len: {len(labels_dict.keys())}")
    # train_generator = MotionDataGenerator(partition['train'], labels_dict, **params)
    # validation_generator = MotionDataGenerator(partition['validation'], labels_dict, **params)
    effort_network = EffortNetwork(two_d_conv=False, model_num=1)
    # effort_network.model.fit_generator(generator=train_generator,
    #                                    validation_data=validation_generator,
    #                                    use_multiprocessing=True,
    #                                    workers=4)
    train_generator = generator(partition['train'], labels_dict)
    validation_generator = generator(partition['validation'], labels_dict)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='auto')
    effort_network.model.fit(generator(partition['train'], labels_dict),
                             validation_data=validation_generator, validation_steps=100, epochs=100,
                             steps_per_epoch=100, callbacks=[early_stopping])

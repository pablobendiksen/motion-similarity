import keras
import random
import os.path
import numpy as np
from glob import glob
import conf
#keras.utils.Sequence
class MotionDataGenerator(keras.utils.Sequence):
    def __init__(self, list_idxs, labels, batch_size=conf.batch_size, batch_dim=(100, 91), shuffle=True):
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_idxs = list_idxs
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.list_idxs)

    def __len__(self):
        return int(np.ceil(len(self.list_idxs) / self.batch_size))

    def __getitem__(self, index):
        # single batch fetching
        batch_ids = self.list_idxs[index * self.batch_size:(index + 1) * self.batch_size]
        batch_features = np.empty((self.batch_size, *self.batch_dim))
        batch_labels = np.empty((self.batch_size, 4))
        for i, id in enumerate(batch_ids):
            data_dir = conf.all_exemplars_folder_3
            path = glob(os.path.join(data_dir, f'*_{id}.npy'))
            if len(path) != 1:
                assert False, f"Error for id {id}, found path for exemplar must be unique — {path}!"
            batch_features = np.zeros((conf.batch_size, 100, 91))
            batch_labels = np.zeros((conf.batch_size, 4))
            batch_features[i] = np.load(path[0])
            batch_labels[i] = self.labels[id]
        return batch_features, batch_labels

    # def generator(self):
    #     batch_nums = self.__len__()
    #     while True:
    #         self.on_epoch_end()
    #         for index in range(batch_nums):
    #             # single batch fetching
    #             batch_ids = self.list_idxs[index * self.batch_size:(index + 1) * self.batch_size]
    #             batch_features = np.empty((self.batch_size, *self.batch_dim))
    #             batch_labels = np.empty((self.batch_size, 4))
    #             for i, id in enumerate(batch_ids):
    #                 data_dir = conf.all_exemplars_folder_3
    #                 path = glob(os.path.join(data_dir, f'*_{id}.npy'))
    #                 if len(path) != 1:
    #                     assert False, f"Error for id {id}, found path for exemplar must be unique — {path}!"
    #                 batch_features = np.zeros((conf.batch_size, 100, 91))
    #                 batch_labels = np.zeros((conf.batch_size, 4))
    #                 batch_features[i] = np.load(path[0])
    #                 batch_labels[i] = self.labels[id]
    #             yield batch_features, batch_labels

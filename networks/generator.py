import keras
import os.path
import numpy as np
from glob import glob
import conf

class MotionDataGenerator(keras.utils.Sequence):
    def __init__(self, list_idxs, labels, batch_size=conf.batch_size, dim=(conf.batch_size, 40, 91), shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_idxs = list_idxs
        self.shuffle = shuffle
        self.indices = np.arange(len(self.list_idxs))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_idxs))
        if self.shuffle is True:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.floor(len(self.list_idxs)) / self.batch_size)

    def __getitem__(self, index):
        # single batch fetching
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_batch_idxs = [self.list_idxs[i] for i in indexes]
        X, y = self._data_generation(list_batch_idxs)

    def _data_generation(self, list_batch_idxs):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 4))

        for i, idx in enumerate(list_batch_idxs):
            data_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/data_tmp"
            path = glob(os.path.join(data_dir, f'*_{idx}.npy'))
            X[i] = np.load(path[0])
            y[i] = self.labels[idx]
            # print(X.shape)
            # print(y.shape)
        return X, y
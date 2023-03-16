import keras
import random
import concurrent
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
        def _load_sample(i, idx):
            print(f"sample: {i}")
            path = glob(os.path.join(conf.all_exemplars_folder_3, f'*_{idx}.npy'))
            if len(path) != 1:
                assert False, f"Error for id {idx}, found path for exemplar must be unique — {path}!"
            batch_features[i] = np.load(path[0])
            batch_labels[i] = self.labels[idx]

        # single batch fetching
        batch_ids = self.list_idxs[index * self.batch_size:(index + 1) * self.batch_size]
        batch_features = np.empty((self.batch_size, *self.batch_dim))
        batch_labels = np.empty((self.batch_size, 4))
        # print(f"labels: {batch_labels}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_load_sample, i, idx) for i, idx in enumerate(batch_ids)]
        f_count = 1
        for future in concurrent.futures.as_completed(futures):
            # handle any exceptions that occurred during execution
            try:
                future.result()
                print(f"future cnt: {f_count}")
                f_count+=1
            except Exception as e:
                print(e)
        # concurrent.futures.wait(futures)
        # for i, id in enumerate(batch_ids):
        #     path = glob(os.path.join(conf.all_exemplars_folder_3, f'*_{id}.npy'))
        #     if len(path) != 1:
        #         assert False, f"Error for id {id}, found path for exemplar must be unique — {path}!"
        #     batch_features[i] = np.load(path[0])
        #     batch_labels[i] = self.labels[id]
        return batch_features, batch_labels



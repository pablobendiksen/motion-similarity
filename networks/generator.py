import concurrent
import os.path
import numpy as np
from glob import glob
import conf


class MotionDataGenerator:
    def __init__(self, list_batch_ids, labels, batch_size=conf.batch_size, batch_dim=(100, 91), shuffle=True):
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_batch_ids = list_batch_ids
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.list_batch_ids)

    def get_num_batches(self):
        return len(self.list_batch_ids)

    def generator(self):

        def _load_batch(i, idx):
            # single batch fetching
            path = glob(os.path.join(conf.all_exemplars_folder_3, f'*_{idx}.npy'))
            if len(path) != 1:
                assert False, f"Error for id {idx}, found path for batch must be unique — {path}!"
            return np.load(path[0]), self.labels[idx]

        def read_async_batch_files(subset_batch_ids):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(_load_batch, i, idx) for i, idx in enumerate(subset_batch_ids)]
                return [fut.result() for fut in futures]

        # read-in batch_group_size batches at a time
        batch_group_size = 16
        num_batches = self.get_num_batches()
        while True:
            self.on_epoch_end()
            for batch_group_idx in range(0, num_batches, batch_group_size):
                group_upper_lim = min(batch_group_idx+batch_group_size, num_batches)
                grouped_batches = read_async_batch_files(self.list_batch_ids[batch_group_idx:group_upper_lim])
                for batch in grouped_batches:
                    yield batch[0], batch[1]



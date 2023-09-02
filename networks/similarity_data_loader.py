import keras
import tensorflow as tf
import concurrent
import random
import os.path
import numpy as np
from glob import glob
import conf
from src.organize_synthetic_data import singleton_batches


class SimilarityDataLoader(keras.utils.Sequence):
    def __init__(self, dict_similarity_exemplars, shuffle=True):
        self.exemplar_dim = conf.similarity_exemplar_dim
        self.batch_size = conf.similarity_batch_size
        self.shuffle = shuffle
        self.dict_similarity_exemplars = dict_similarity_exemplars
        print(f"num classes in dict: {len(self.dict_similarity_exemplars.keys())}")
        self.class_indexes, self.list_class_tuples = zip(
            *[(index + 1, value) for index, value in enumerate(list(self.dict_similarity_exemplars.keys()))])
        self.num_classes = len(self.class_indexes)
        # self.num_batches = len(self.dict_similarity_exemplars[self.list_class_tuples[0]])
        self.num_batches = len(list(self.dict_similarity_exemplars.values())[0])
        self.exemplar_idx = random.randint(1, self.num_batches)

    def unison_shuffling(self):
        # generate random batch num index, to be applied to all dict class lists, and shuffle
        # class + class_idx order (within batch shuffling)
        self.exemplar_idx = random.randint(1, self.num_batches)
        p = np.random.permutation(self.num_classes)
        self.class_indexes = self.class_indexes[p]
        self.list_class_tuples = self.list_class_tuples[p]

    def on_epoch_end(self):
        if self.shuffle is True:
            self.unison_shuffling()

    def __len__(self):
        # num_batches
        return self.num_batches

    def __getitem__(self, index):
        # tensor: (56, 100, 91), tensor (64, 1)
        batch_features = tf.stack([self.dict_similarity_exemplars[class_tuple][self.exemplar_idx] for class_tuple in
                                   self.list_class_tuples])
        batch_labels = tf.constant(self.class_indexes)
        assert tf.shape(batch_features) == (56, 100, 91), (f"Similarity batch error; expected shape: {(56, 100, 91)}, "
                                                           f"actual shape{tf.shape(batch_features)}")
        return batch_features, batch_labels

import keras
import tensorflow as tf
import concurrent
import random
import os.path
import numpy as np
from glob import glob


class SimilarityDataLoader(keras.utils.Sequence):
    def __init__(self, list_similarity_dicts, config, shuffle=False):
        print("initializing similarity data loader")
        self.config = config
        self.exemplar_dim = self.config.similarity_exemplar_dim
        self.batch_size = self.config.similarity_batch_size
        self.shuffle = shuffle
        self.dict_similarity_exemplars = {}
        self.class_indexes = []
        # self.class_indexes, _unused_var = zip(
        #     *[(index + 1, value) for index, value in enumerate(list(list_similarity_dicts[0].keys()))])
        self.list_class_tuples = []
        for i, similarity_dict in enumerate(list_similarity_dicts):
            for j, (class_id, value) in enumerate(similarity_dict.items()):
                new_key = (i, class_id)
                self.dict_similarity_exemplars[new_key] = value
                self.list_class_tuples.append(new_key)
                self.class_indexes.append(j)
        # Each key is a tuple containing the dictionary identifier and the original key (state or drive index id).
        self.num_classes = len(self.class_indexes)/3
        print(f"SimilarityDataLoader: num classes: {self.num_classes}")

        # self._num_batches = len(self.list_class_tuples) // self.batch_size
        # if len(self.list_class_tuples) % self.batch_size != 0:
        #     self._num_batches += 1

        self._num_batches = len(self.dict_similarity_exemplars[next(iter(self.dict_similarity_exemplars.keys()))][0])
        print(f"SimilarityDataLoader: num batches: {self._num_batches}")
        # self.exemplar_idx = random.randint(0, self.num_batches-1)
        self.exemplar_idx = 0

    def unison_shuffling(self):
        # generate random batch num index, to be applied to all dict class lists, and unison shuffle
        # class and class_idx order (a within batch shuffling)
        self.exemplar_idx = random.randint(1, self._num_batches - 1)
        p = list(np.random.permutation(self.num_classes))
        # the tight coupling between class_indexes and list_class_tuples allows for class index (batch label) to
        # relevant class exemplar (batch feature) mapping
        self.class_indexes = [self.class_indexes[i] for i in p]
        self.list_class_tuples = [self.list_class_tuples[i] for i in p]

    def on_epoch_end(self):
        if self.shuffle is True:
            self.unison_shuffling()

    def __len__(self):
        # num_batches
        return self._num_batches

    def __getitem__(self, index):
        batch_features = tf.stack([self.dict_similarity_exemplars[class_tuple][0] for class_tuple in
                                   self.list_class_tuples])
        batch_labels = tf.constant(self.class_indexes)
        # batch_features = tf.stack([self.dict_similarity_exemplars[class_tuple][0] for class_tuple in
        #                            self.list_class_tuples])
        # batch_labels = tf.constant(self.class_indexes)
        return batch_features, batch_labels

    # def __getitem__(self, index):
    #     # batch features shape: tensor: [ 56 100  91],  batch labels shape: tensor: [56]
    #     # if not conf.bool_fixed_neutral_embedding:
    #     #     pass
    #     batch_features = tf.stack([self.dict_similarity_exemplars[class_tuple][self.exemplar_idx] for class_tuple in
    #                                self.list_class_tuples])
    #     batch_labels = tf.constant(self.class_indexes)
    #     # tf.debugging.assert_shapes([(batch_features, (conf.similarity_batch_size, conf.similarity_exemplar_dim[0],
    #     #                                               conf.similarity_exemplar_dim[1]))],
    #     #                            message="Similarity batch feature erroneous shape}")
    #     return batch_features, batch_labels

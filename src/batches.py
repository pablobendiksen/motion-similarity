import conf
import numpy as np
import tensorflow as tf
import pickle
import math


class Batches:
    # organize batch related aliases and functionality here
    def __init__(self):
        self.batch_size = conf.batch_size_efforts_network
        self.exemplar_efforts_dim = conf.exemplar_dim_effort_network
        self.sample_idx = 0
        self.batch_idx = 0
        self.tmp_exemplar = np.zeros(conf.exemplar_dim_effort_network)
        self.state_drive_exemplar_idx = None
        self.current_batch = {0: []}
        self.list_batch_efforts = []
        self.dict_efforts_labels = {0: np.zeros((self.batch_size, conf.num_efforts))}
        self.dict_similarity_exemplars = self._generate_similarity_classes_exemplars_dict()
        self.sliding_window_start_index = conf.time_series_size

    def append_efforts_batch_and_labels(self, exemplar):
        self.list_batch_efforts.append(exemplar[0][0:conf.num_efforts])
        # drop anim name column
        exemplar = np.delete(exemplar, slice(5), axis=1)
        print(f"EXEMPLAR SHAPE WHEN STORING: {exemplar.shape}")
        self.current_batch[self.batch_idx].append(exemplar)
        self.sample_idx += 1

    def store_efforts_batch(self):
        self.dict_efforts_labels[self.batch_idx] = np.array(self.list_batch_efforts)
        assert len(self.dict_efforts_labels[self.batch_idx]) == 64, f"Len of efforts dict at batch idx " \
                                                                    f"{self.batch_idx} is {len(self.dict_efforts_labels[self.batch_idx])}"

        batch_array = np.array(self.current_batch[self.batch_idx])
        np.save(conf.exemplars_dir + 'batch_' + str(self.batch_idx) + '.npy',
                batch_array)
        print(
            f"Stored batch num {self.batch_idx}. Size: {batch_array.shape}.  exemplar count:"
            f" {self.sample_idx}")
        self.tmp_exemplar = batch_array[0]
        self.batch_idx += 1
        self.current_batch[self.batch_idx] = []
        self.list_batch_efforts = []
        self.dict_efforts_labels[self.batch_idx] = np.zeros((self.batch_size, conf.num_efforts))

    @staticmethod
    def append_to_end_file_exemplar(exemplar, make_whole_exemplar=False):
        target_row_num = conf.time_series_size
        last_row = exemplar[-1]
        if make_whole_exemplar:
            repeats = target_row_num
        else:
            repeats = target_row_num - exemplar.shape[0]
        extended_rows = np.tile(last_row, (repeats, 1))
        print(f"make_whole_exemplar: {make_whole_exemplar}, concatenated matrix shape:"
              f"{np.concatenate((exemplar, extended_rows), axis=0)}")
        return np.concatenate((exemplar, extended_rows), axis=0)

    def extend_final_batch(self, exemplar):
        # exemplar = np.delete(end_directory_exemplar, slice(5), axis=1)
        last_row = exemplar[-1]
        repeats = conf.time_series_size
        while len(self.current_batch[self.batch_idx]) < conf.batch_size_efforts_network:
            new_exemplar = np.tile(last_row, (repeats, 1))
            self.sample_idx += 1
            self.append_efforts_batch_and_labels(new_exemplar)
        self.store_efforts_batch()

    def store_effort_labels_dict(self):
        self.dict_efforts_labels.pop(self.batch_idx)
        self.current_batch.pop(self.batch_idx)
        self.batch_idx -= 1
        print(f"storing dict_efforts_labels with len {len(self.dict_efforts_labels)} with dict of batches size"
              f" {len(self.current_batch)}")
        with open(conf.exemplars_dir + '/labels_dict.pickle', 'wb') as handle:
            pickle.dump(self.dict_efforts_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"storing {len(self.dict_efforts_labels.keys())} batch labels")
        conf.exemplar_num = self.sample_idx

    def store_similarity_labels_exemplars_dict(self):
        with open(conf.exemplars_dir + conf.similarity_dict_file_name, 'wb') as handle:
            pickle.dump(self.dict_similarity_exemplars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"storing {len(self.dict_similarity_exemplars.keys())} similarity exemplars")

    def move_tuple_to_similarity_dict_front(self, key=(0, 0, 0, 0)):
        # move neutral tuple to front of dict
        neut_value = self.dict_similarity_exemplars.pop(key)
        self.dict_similarity_exemplars = {key: neut_value} | self.dict_similarity_exemplars

    def convert_exemplar_np_arrays_to_tensors(self):
        for state_drive, inner_list in self.dict_similarity_exemplars.items():
            self.dict_similarity_exemplars[state_drive] = [tf.convert_to_tensor(exemplar, dtype=tf.float32) for
                                                           exemplar in inner_list]
        print(f"converted {len(self.dict_similarity_exemplars[(0, 0, 0, 0)])} k,v similarity exemplars to type: "
              f"{type(self.dict_similarity_exemplars[(0, 0, 0, 0)][0])}")

    # def append_similarity_class_exemplar(self, state_drive, exemplar):
    #     # include efforts but not anim
    #     exemplar = np.delete(exemplar, 4, axis=1)
    #     self.dict_similarity_exemplars[state_drive][self.state_drive_exemplar_idx] = exemplar
    #     self.state_drive_exemplar_idx += 1

    def append_similarity_class_exemplar(self, state_drive, exemplar):
        # include efforts but not anim
        exemplar = np.delete(exemplar, 4, axis=1)
        self.dict_similarity_exemplars[state_drive].append(exemplar)
        self.state_drive_exemplar_idx += 1

    def print_len_dict_similarity_exemplars(self):
        print("Printing len of dict_similarity_exemplars")
        for key in self.dict_similarity_exemplars:
            print(f"Key: {key}. Len: {len(self.dict_similarity_exemplars[key])}")

    # def balance_similarity_classes(self):
    #     max_inner_keys = max(len(inner_dict) for inner_dict in self.dict_similarity_exemplars.values())
    #     for state_drive, inner_dict in self.dict_similarity_exemplars.items():
    #         length_inner_key = len(inner_dict)
    #         num_inner_keys = length_inner_key
    #         # for toy data lacking exemplars to multiple classes
    #         if length_inner_key == 0:
    #             num_inner_keys = 1
    #             # {exemplar idx: exemplar}
    #             inner_dict[0] = self.tmp_exemplar
    #         # convert num_inner_keys to num_inner_list_elems
    #         if num_inner_keys < max_inner_keys:
    #             # perform pop of inner_list
    #             last_exemplar = inner_dict[num_inner_keys - 1]
    #             for i in range(num_inner_keys, max_inner_keys):
    #                 new_exemplar = self.append_to_end_file_exemplar(last_exemplar, make_whole_exemplar=True)
    #                 # inner_list.append(new_exemplar)
    #                 inner_dict[i] = new_exemplar
    #     list_dict_lens = [len(inner_dict) for inner_dict in self.dict_similarity_exemplars.values()]
    #     assert all(dict_len == list_dict_lens[0] for dict_len in list_dict_lens), "Classes have different exemplar " \
    #                                                                               "counts"

    def balance_similarity_classes(self):
        max_inner_list_len = max(len(inner_list) for inner_list in self.dict_similarity_exemplars.values())
        inner_list_lens = []
        for state_drive, inner_list in self.dict_similarity_exemplars.items():
            length_inner_list = len(inner_list)
            inner_list_lens.append(length_inner_list)
            num_inner_list_elems = length_inner_list
            # for toy data lacking exemplars to multiple classes
            if length_inner_list == 0:
                num_inner_list_elems = 1
                # {exemplar idx: exemplar}
                inner_list[0] = self.tmp_exemplar
            # convert num_inner_list_elems to max_inner_list_len
            if num_inner_list_elems < max_inner_list_len:
                # perform pop of inner_list
                last_exemplar = inner_list[-1]
                for i in range(num_inner_list_elems, max_inner_list_len):
                    new_exemplar = self.append_to_end_file_exemplar(last_exemplar, make_whole_exemplar=True)
                    inner_list.append(new_exemplar)
                print(f"Newly appended similarity class exemplar list is of len: {len(inner_list)}")
        assert all(dict_len == inner_list_lens[0] for dict_len in inner_list_lens), "Classes have different exemplar " \
                                                                                  "counts"

    @staticmethod
    def _generate_similarity_classes_exemplars_dict():
        """
        Generate Dict for mapping Similarity class to exemplar number and data
        Returns:
            dict_similarity_exemplars: Dictionary. {similarity_class: {}}

        Once populated, using training dataset: stylized motion effort_walking_105_34_552.bvh, the dict will look like:
        {(0, 0, 0, 0): [tensor([100, 91]), tensor([100, 91]), ... ], (0, 0, 0, 1): ...} for 57 keys, each with
        a list of 37 exemplars of size (100, 91). This makes each class span the length of about two minutes and the
        dictionary size of 2 GB.

        """

        def generate_states_and_drives():
            def convert_to_nary(tuple_index, values_per_effort, efforts_per_tuple):
                effort_tuple = []
                zeroes_counter = 0
                for _ in range(efforts_per_tuple):
                    tuple_index, remainder = divmod(tuple_index, values_per_effort)
                    if remainder == 1:
                        zeroes_counter += 1
                    effort_tuple.append(effort_vals[remainder])
                if zeroes_counter == 2 or zeroes_counter == 1:
                    states_and_drives.append(effort_tuple)

            states_and_drives = []
            effort_vals = [-1, 0, 1]
            values_per_effort = 3
            efforts_per_tuple = 4
            tuple_index = 0
            while (tuple_index < math.pow(values_per_effort, 4)):
                tuple_index += 1
                convert_to_nary(tuple_index, values_per_effort, efforts_per_tuple)
            # include the neutral tuple
            states_and_drives.append([0, 0, 0, 0])
            return states_and_drives

        states_and_drives = generate_states_and_drives()
        print(f"states_and_drives: {states_and_drives}")
        # return {tuple(state_drive): {} for state_drive in states_and_drives}
        return {tuple(state_drive): [] for state_drive in states_and_drives}

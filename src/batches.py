import conf
import numpy as np
import pickle
import math

class Batches:
    # organize batch related aliases and functionality here
    def __init__(self):
        self.batch_size = conf.batch_size_efforts_network
        self.exemplar_efforts_dim = conf.exemplar_dim_effort_network
        self.sample_idx = 0
        self.batch_idx = 0
        self.state_drive_exemplar_idx = None
        self.current_batch = []
        self.dict_efforts_labels = {self.batch_idx: []}
        self.dict_similarity_exemplars = self._generate_similarity_classes_exemplars_dict()
        self.sliding_window_start_index = conf.time_series_size

    def append_batch_and_labels(self, exemplar):
        self.dict_efforts_labels[self.batch_idx].append(exemplar[0][0:conf.num_efforts])
        # drop labels and anim columns
        exemplar = np.delete(exemplar, slice(5), axis=1)
        # print(f"EXEMPLAR SHAPE WHEN STORING: {exemplar.shape}")
        self.current_batch.append(exemplar)
        self.sample_idx += 1

    @staticmethod
    def append_to_end_file_exemplar(exemplar, make_whole_exemplar=False):
        target_row_num = conf.time_series_size
        last_row = exemplar[-1]
        if make_whole_exemplar:
            repeats = target_row_num
        else:
            repeats = target_row_num - exemplar.shape[0]
        extended_rows = np.tile(last_row, (repeats, 1))
        return np.concatenate((exemplar, extended_rows), axis=0)

    def extend_final_batch(self, end_directory_exemplar):
        exemplar = np.delete(end_directory_exemplar, slice(5), axis=1)
        last_row = exemplar[-1]
        repeats = conf.time_series_size
        while len(self.current_batch) < conf.batch_size_efforts_network:
            new_exemplar = np.tile(last_row, (repeats, 1))
            self.sample_idx += 1
            self.current_batch.append(new_exemplar)
        self.store_batch()

    def store_batch(self):
        self.dict_efforts_labels[self.batch_idx] = np.array(self.dict_efforts_labels[self.batch_idx])
        motions = np.array(self.current_batch)
        np.save(conf.exemplars_dir + 'batch_' + str(self.batch_idx) + '.npy',
                motions)
        print(f"stored batch num {self.batch_idx}. Size: {motions.shape}.  exemplar count: {self.sample_idx}")
        self.current_batch = []
        self.batch_idx += 1
        self.dict_efforts_labels[self.batch_idx] = []

    def store_effort_labels_dict(self):
        self.dict_efforts_labels.pop(self.batch_idx)
        self.batch_idx -= 1
        with open(conf.exemplars_dir + '/labels_dict.pickle', 'wb') as handle:
            pickle.dump(self.dict_efforts_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"storing {len(self.dict_efforts_labels.keys())} batch labels")
        conf.exemplar_num = self.sample_idx

    def append_similarity_class_exemplar(self, state_drive, exemplar):
        # include efforts but not anim
        exemplar = np.delete(exemplar, 4, axis=1)
        self.dict_similarity_exemplars[state_drive][self.state_drive_exemplar_idx] = exemplar
        self.state_drive_exemplar_idx += 1

    def print_len_dict_similarity_exemplars(self):
        print("Printing len of dict_similarity_exemplars")
        for key in self.dict_similarity_exemplars:
            print(f"Key: {key}. Len: {len(self.dict_similarity_exemplars[key])}")

    def balance_similarity_classes(self):
        max_inner_keys = max(len(inner_dict) for inner_dict in self.dict_similarity_exemplars.values())
        for state_drive, inner_dict in self.dict_similarity_exemplars.items():
            num_inner_keys = len(inner_dict)
            if num_inner_keys < max_inner_keys:
                print(f"{inner_dict=}")
                last_exemplar = inner_dict[num_inner_keys - 1]
                for i in range(num_inner_keys, max_inner_keys):
                    new_exemplar = self.append_to_end_file_exemplar(last_exemplar, make_whole_exemplar=True)
                    inner_dict[i] = new_exemplar

    @staticmethod
    def _generate_similarity_classes_exemplars_dict():
        """Generate Dict for mapping Similarity class to exemplar number and data
        Returns:
            dict_similarity_exemplars: Dictionary. {similarity_class: {}}
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
                if zeroes_counter == 2 or zeroes_counter == 1 or zeroes_counter == 4:
                    states_and_drives.append(effort_tuple)

            states_and_drives = []
            effort_vals = [-1, 0, 1]
            values_per_effort = 3
            efforts_per_tuple = 4
            tuple_index = 0
            while (tuple_index < math.pow(values_per_effort, 4)):
                tuple_index += 1
                convert_to_nary(tuple_index, values_per_effort, efforts_per_tuple)
            return states_and_drives

        states_and_drives = generate_states_and_drives()
        print(f"states_and_drives: {states_and_drives}")
        return {tuple(state_drive): {} for state_drive in states_and_drives}
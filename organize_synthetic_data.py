import os

import numpy as np

from pymo.parsers import BVHParser

from pymo.viz_tools import *
from sklearn.preprocessing import StandardScaler
from pymo.preprocessing import *
from os import path
import conf
from sklearn.pipeline import Pipeline
import seaborn as sns
import pickle
import warnings
import random
import math

anim_ind = {'WALKING': 0, 'POINTING': 1, 'PICKING': 2, 'WAVING': 3, 'THROWING': 4, 'AIMING': 5, 'JUMPING': 6,
            'RUNNING': 7}
parser = BVHParser()

#TODO: Manage complexity of this file by creating EffortBatches class
class EffortBatches():
    def __init__(self):
        # organize batch related aliases and functionality here
        self.batch_size = conf.batch_size_efforts_predictor
        self.exemplar_dim = (100, 87)
        self.sample_idx = 0
        self.batch_idx = 0
        self.current_batch = None
        self.labels_dict = None
        self.sliding_window_start_index = conf.time_series_size
        self.sliding_window_end_index = None

    def _store_batch(self):
        self.labels_dict[self.batch_idx] = np.array(self.labels_dict[self.batch_idx])
        motions = np.array(self.current_batch)
        # np.save(conf.exemplars_dir + 'batch_' + str(batch_idx) + '.npy',
        #         motions)
        # print(f"stored batch num {batch_idx}. Size: {motions.shape}.  exemplar count: {sample_idx}")
        # batch = []
        # batch_idx += 1
        # labels_dict[batch_idx] = []


def generate_similarity_classes_exemplars_dict():
    """Generate Dict for mapping Similarity class to exemplar number and data
    Returns:
        dict_similarity_exemplars: Dictionary. {similarity_class: {}}
    """

    # effort states and drives comprise the classes for the similarity network
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
        return states_and_drives

    states_and_drives = generate_states_and_drives()
    print(states_and_drives)
    return {tuple(state_drive): {} for state_drive in states_and_drives}


def visualize(file_bvh):
    parsed_data = parser.parse(file_bvh)
    data_pipe = Pipeline([
        # ('param2', ConverterToRightHandedCoordinates()),
        ('param1', MocapParameterizer('expmap')),
        # ('down', DownSampler(4)),  # downsample to 30fps
        # ('stdscale', ListStandardScaler())
    ])
    data = parsed_data
    mp = MocapParameterizer('position')
    positions = mp.transform([data])[0]

    nb_play_mocap(positions, 'pos',
                  scale=2, camera_z=800, frame_time=1 / 30,
                  base_url='pymo/mocapplayer/playBuffer2.html')


def clear_file(file):
    # removes character name from the file
    # read input file
    fin = open(file, "rt")
    # read file contents to string
    data = fin.read()
    # replace all occurrences of the required string
    data = data.replace('Carl:', '')
    # close the input file
    fin.close()
    # open the input file in write mode
    fin = open(file, "wt")
    # overwrite the input file with the resulting data
    fin.write(data)
    # close the file
    fin.close()


def prep_all_data_for_training(anim_name=None, rotations=True, velocities=False):
    # all joints (corresponding to 3 columns each [Z, X,Y dimensions]) now have absolute positions
    def _preprocess_pipeline(parsed_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_pipe_expmap = Pipeline(steps=[
                ('param', MocapParameterizer('expmap')),
                ('np', Numpyfier()),
                ('down', DownSampler(4))
            ])
        return data_pipe_expmap.fit_transform([parsed_data])[0]

    def _get_standardized_rotations(data_expmaps):
        # generate z-scores for all values by means of sklearn StandardScaler (i.e., standardize!)
        data_expmaps = _z_score_generator(data_expmaps)
        return data_expmaps

    def _get_standardized_velocities(data_velocities):
        # calculate velocities
        # needed for proper broadcasting of following step
        frame_rate_array = np.tile(bvh_frame_rate.pop(), (data_velocities.shape[0] - 1, data_velocities.shape[1]))
        # now calculate velocities from positions
        data_velocities[1:] = (data_velocities[1:, :] - data_velocities[:-1, :]) / frame_rate_array
        data_velocities[0] = 0
        # generate z-scores for all values by means of sklearn StandardScaler (i.e., standardize!)
        data_velocities = _z_score_generator(data_velocities)
        return data_velocities

    def _z_score_generator(np_array):
        scaler = StandardScaler()
        scaler = scaler.fit(np_array)
        np_array = scaler.transform(np_array)
        return np_array

    def create_corr_matrix(np_array, name):
        print("NAME:", name)
        dataframe = pd.DataFrame(np_array)
        print("DATAFRAME :", dataframe)
        a = dataframe.corr()
        ax = sns.heatmap(
            a,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        plt.title(name)
        plt.show()
        # ------------

    def apply_moving_window(file_data, sample_idx, labels_dict, batch, batch_idx, exemplar_state_or_drive_count):
        """helper function for concat_all_data_as_np()
            exemplar: np.array
            anim: str
                animation name
            idx: int
                exemplar index
            labels_dict: dictionary
                Dictionary linking file_idx to label
            batch: list
                empty list for housing num_batches exemplars (np.arrays)"""

        def store_batch(batch, batch_idx):
            labels_dict[batch_idx] = np.array(labels_dict[batch_idx])
            motions = np.array(batch)
            np.save(conf.exemplars_dir + 'batch_' + str(batch_idx) + '.npy',
                    motions)
            print(f"stored batch num {batch_idx}. Size: {motions.shape}.  exemplar count: {sample_idx}")
            batch = []
            batch_idx += 1
            labels_dict[batch_idx] = []
            return labels_dict, sample_idx, batch, batch_idx

        def append_to_end_file_exemplar(end_file_exemplar):
            target_row_num = conf.time_series_size
            last_row = end_file_exemplar[-1]
            repeats = target_row_num - end_file_exemplar.shape[0]
            extended_rows = np.tile(last_row, (repeats, 1))
            end_file_exemplar = np.concatenate((end_file_exemplar, extended_rows), axis=0)
            return end_file_exemplar

        def extend_final_batch(end_directory_exemplar, idx, batch):
            while len(batch) < conf.batch_size_efforts_predictor:
                last_row = end_directory_exemplar[-1]
                repeats = conf.time_series_size
                new_exemplar = np.tile(last_row, (repeats, 1))
                idx += 1
                batch.append(new_exemplar)
            return batch, idx

        start_index = conf.time_series_size
        end_index = file_data.shape[0]
        for i in range(start_index, end_index + conf.window_delta, conf.window_delta):
            indices = range(i - conf.time_series_size, i)
            # end of file corner case correction
            if i > end_index:
                indices = range(i - conf.time_series_size, end_index)
                end_file_exemplar = append_to_end_file_exemplar(file_data[indices])
                labels_dict[batch_idx].append(end_file_exemplar[0][0:conf.num_efforts])
                exemplar_tmp = np.delete(end_file_exemplar, slice(5), axis=1)
                batch.append(exemplar_tmp)
                if bool_state_or_drive:
                    print("In")
                    # include efforts but not anim
                    exemplar_tmp = np.delete(end_file_exemplar, 4, axis=1)
                    dict_similarity_exemplars[tuple_effort_list][exemplar_state_or_drive_count] = exemplar_tmp
                    exemplar_state_or_drive_count = exemplar_state_or_drive_count + 1
                sample_idx += 1
                # end of directory file corner case correction
                if f == filenames[-1]:
                    batch, sample_idx = extend_final_batch(exemplar_tmp, sample_idx, batch)
                    print(f"Final batch len: {len(batch)}")
            else:
                labels_dict[batch_idx].append(file_data[indices[0]][0:conf.num_efforts])
                # drop labels and anim columns
                exemplar_tmp = np.delete(file_data[indices], slice(5), axis=1)
                print(f"exemplar size: {exemplar_tmp.shape}")
                batch.append(exemplar_tmp)
                sample_idx += 1
                if bool_state_or_drive:
                    print("In")
                    exemplar_tmp = np.delete(file_data[indices], 4, axis=1)
                    dict_similarity_exemplars[tuple_effort_list][exemplar_state_or_drive_count] = exemplar_tmp
                    exemplar_state_or_drive_count = exemplar_state_or_drive_count + 1
            if len(batch) == conf.batch_size_efforts_predictor:
                labels_dict, sample_idx, batch, batch_idx = store_batch(batch, batch_idx)

        return labels_dict, sample_idx, batch, batch_idx

    sample_idx = 0  # unique for each exemplar
    labels_dict = {}  # populated across all files
    dict_similarity_exemplars = generate_similarity_classes_exemplars_dict()  # map user study efforts to exemplar info
    batch = []
    batch_idx = 0
    labels_dict[0] = []
    dir = conf.bvh_subsets_dir
    bvh_counter = 0
    bvh_frame_rate = set()
    filenames = os.listdir(dir)
    tuple_effort_lists = []
    bool_state_or_drive = False
    exemplar_state_or_drive_count = 0
    for f in filenames:
        name = path.splitext(f)[0]  # exclude extension bvh by returning the root
        name_split = name.split('_') # get effort values from the file name
        anim = name_split[0]
        f_full_path = dir + f
        if anim_name is None or str.upper(anim_name) == str.upper(anim):
            # extract efforts
            efforts_list = [float(p) for p in name.split('_')[-4:]]
            tuple_effort_list = tuple(efforts_list)
            if tuple_effort_list in dict_similarity_exemplars.keys():
                print(f"found tuple: {tuple_effort_list}")
                bool_state_or_drive = True
                exemplar_state_or_drive_count = 1
                tuple_effort_lists.append(tuple_effort_list)
                dict_similarity_exemplars[tuple_effort_list] = {}
            else:
                bool_state_or_drive = False
            clear_file(f_full_path)  # remove the : from the file
            parsed_data = parser.parse(f_full_path) # parsed file of type pymo.data.MocapData
            bvh_frame_rate.add(parsed_data.framerate)
            if len(bvh_frame_rate) > 1:
                fr = bvh_frame_rate.pop()
                print(f"frame rate of: {fr} found for bvh file index {bvh_counter}.\nfile discarded")
                continue
            # ensure consistent frame rate across motion files (otherwise velocities miscalculated)
            assert len(bvh_frame_rate) == 1, f"More than one frame rate present!!! {bvh_frame_rate}"
            if rotations and velocities:
                file_name = 'data/all_synthetic_motions_velocities_effort.csv'
                data_expmaps = _preprocess_pipeline(parsed_data)
                data_expmaps = _get_standardized_rotations(data_expmaps)
                # uncomment only to view corr_matrix of file
                # create_corr_matrix(data_expmaps, name)
                data_expmaps = data_expmaps[:, 3:]
                data_velocities = _get_standardized_velocities(parsed_data)
                # stack expmap angles for all joints horizontally to data_velocities
                data = np.hstack((data_velocities, data_expmaps))
            elif not rotations and velocities:
                file_name = 'data/all_synthetic_motions_velocities_only_effort.csv'
                data = _get_standardized_velocities(parsed_data)
            else:
                data = _get_standardized_rotations(parsed_data)
                file_name = 'data/all_synthetic_motions_effort.csv'
            bvh_counter += 1
            if data.shape[0] < conf.time_series_size:
                assert False, f"Preprocessed file too small- {data.shape[0]} - relative to exemplar size -" \
                              f" {conf.time_series_size}"
            f_rep = np.tile(efforts_list, (data.shape[0], 1))
            # append anim as an additional column
            a_rep = np.tile(anim_ind[str.upper(anim)], (data.shape[0], 1))
            # animation name is fifth column
            file_data = np.concatenate((a_rep, data), axis=1)
            # append efforts (the first 4 column(s) will be the efforts)
            file_data = np.concatenate((f_rep, file_data), axis=1)
            print(f"file data size: {file_data.shape}")
            labels_dict, sample_idx, batch, batch_idx = apply_moving_window(file_data, sample_idx,
                                                                            labels_dict, batch, batch_idx,
                                                                            exemplar_state_or_drive_count)

    conf.bvh_file_num = bvh_counter
    conf.exemplar_num = sample_idx
    with open(conf.exemplars_dir + '/labels_dict.pickle', 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"storing {len(labels_dict.keys())} labels")
    print(f"len of tuple_effort_lists: {len(tuple_effort_lists)}")
    print(f"{dict_similarity_exemplars=}")


def prepare_data(rotations=True, velocities=False):
    prep_all_data_for_training(rotations=rotations, velocities=velocities)


def load_data(rotations=True, velocities=False):
    if not path.exists(conf.exemplars_dir):
        os.makedirs(conf.exemplars_dir)
        prepare_data(rotations=rotations, velocities=velocities)
    elif not os.listdir(conf.exemplars_dir):
        prepare_data(rotations=rotations, velocities=velocities)

    partition, labels_dict = _load_ids_and_labels()
    return partition, labels_dict


def _load_ids_and_labels(train_val_split=0.8):
    with open(conf.exemplars_dir + 'labels_dict.pickle', 'rb') as handle:
        labels_dict = pickle.load(handle)
    batch_ids_list = list(labels_dict.keys())
    random.shuffle(batch_ids_list)
    train_size = int(train_val_split * len(batch_ids_list))
    test_val_size = int(((1 - train_val_split) * len(batch_ids_list)) / 2)
    partition = {'train': batch_ids_list[:train_size], 'validation': batch_ids_list[train_size:test_val_size],
                 'test': batch_ids_list[-test_val_size:]}
    return partition, labels_dict


def load_data_for_prediction():
    file = 'data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy'
    if path.exists(file):
        data = np.load(file)

    else:
        for i in range(12):
            print(f"this is a test of a for loop: {i}")
        prepare_data()
        data = np.load(file)

    labels = np.ndarray(data.shape)

    new_data_len = data.shape[0] - 1
    for i in range(new_data_len):
        # make labels equal to the original data
        labels[i] = data[i]
        # TODO: prediction
        # labels[i] = data[i+1]
    return data[0:new_data_len, :, :], labels[0:new_data_len, :, :]


# param efforts matched against synthetic_motion array indices and efforts removed from resulting array
def load_effort_animation(animName, efforts):
    name = path.splitext(conf.all_concatenated_motions_file)[0]  # exclude extension csv
    file_name = name + "_" + str.upper(animName) + '.csv'

    motions = np.genfromtxt(file_name, delimiter=',')

    labels = motions[:, 0:conf.num_efforts]

    indices = [i for i, val in enumerate(labels) if np.array_equal(val, efforts)]

    data = np.delete(motions[indices], range(0, conf.num_efforts), axis=1)

    return data


def prepare_comparison_data():
    prep_all_data_for_training("pointing")


if __name__ == "__main__":
    load_data(rotations=True, velocities=False)

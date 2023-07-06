import os

from src.batches import Batches
from pymo.parsers import BVHParser
from pymo.viz_tools import *
from sklearn.preprocessing import StandardScaler
from pymo.preprocessing import *
from os import path
import shutil
import conf
from sklearn.pipeline import Pipeline
import numpy as np
import seaborn as sns
import pickle
import warnings
import random

anim_ind = {'WALKING': 0, 'POINTING': 1, 'PICKING': 2, 'WAVING': 3, 'THROWING': 4, 'AIMING': 5, 'JUMPING': 6,
            'RUNNING': 7}
parser = BVHParser()


def visualize(file_bvh):
    parsed_data = parser.parse(file_bvh)
    data_pipe = Pipeline([
        ('param1', MocapParameterizer('expmap')),
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


def prep_all_data_for_training(rotations=True, velocities=False):
    def _preprocess_pipeline(parsed_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_pipe_expmap = Pipeline(steps=[
                ('param', MocapParameterizer('expmap')),
                ('np', Numpyfier()),
                # ('down', DownSampler(4))
            ])
        return data_pipe_expmap.fit_transform([parsed_data])[0]

    def _get_standardized_rotations(data_expmaps):
        data_expmaps = _z_score_generator(data_expmaps)
        return data_expmaps

    def _get_standardized_velocities(data_velocities):
        # needed for proper broadcasting of following step
        frame_rate_array = np.tile(bvh_frame_rate.pop(), (data_velocities.shape[0] - 1, data_velocities.shape[1]))
        # calculate velocities from positions
        data_velocities[1:] = (data_velocities[1:, :] - data_velocities[:-1, :]) / frame_rate_array
        data_velocities[0] = 0
        # standardize velocities
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

    def apply_moving_window(batches, file_data):
        """helper function for concat_all_data_as_np()
            batches: instance of Batches class
            file_data: np.array comprising exemplar
                animation name"""

        start_index = conf.time_series_size
        end_index = file_data.shape[0]
        for i in range(start_index, end_index + conf.window_delta, conf.window_delta):
            indices = range(i - conf.time_series_size, i)
            # end of file corner case correction
            if i > end_index:
                indices = range(i - conf.time_series_size, end_index)
                exemplar = file_data[indices]
                exemplar = batches.append_to_end_file_exemplar(exemplar)
                batches.append_batch_and_labels(exemplar)
                if f == filenames[-1]:
                    batches.extend_final_batch(exemplar)
            else:
                exemplar = file_data[indices]
                batches.append_batch_and_labels(exemplar)
            if tuple_effort_list in batches.dict_similarity_exemplars.keys():
                batches.append_similarity_class_exemplar(tuple_effort_list, file_data[indices])
            if len(batches.current_batch[batches.batch_idx]) == conf.batch_size_efforts_network:
                batches.store_batch()

    bvh_counter = 0
    bvh_frame_rate = set()
    filenames = os.listdir(conf.bvh_files_dir)
    print(conf.bvh_files_dir)
    print(filenames)
    batches = Batches()
    for f in filenames:
        if f.endswith("bvh"):
            name = path.splitext(f)[0]  # exclude extension bvh by returning the root
            name_split = name.split('_')  # get effort values from the file name
            anim = name_split[0]
            f_full_path = conf.bvh_files_dir + f
            efforts_list = [float(p) for p in name.split('_')[-4:]]
            tuple_effort_list = tuple(efforts_list)
            batches.state_drive_exemplar_idx = 0
            clear_file(f_full_path)  # remove the : from the file
            parsed_data = parser.parse(f_full_path)  # parsed file of type pymo.data.MocapData
            bvh_frame_rate.add(parsed_data.framerate)
            assert len(bvh_frame_rate) == 1, f"More than one frame rate present!!! {bvh_frame_rate}"
            if rotations and velocities:
                file_name = 'data/all_synthetic_motions_velocities_effort.csv'
                data_expmaps = _preprocess_pipeline(parsed_data)
                # remove root joint absolute positions
                data_expmaps = data_expmaps[:, 3:]
                data_velocities = _get_standardized_velocities(parsed_data)
                # stack expmap angles for all joints horizontally to data_velocities
                data = np.hstack((data_velocities, data_expmaps))
            elif not rotations and velocities:
                data_expmaps = _preprocess_pipeline(parsed_data)
                data = _get_standardized_velocities(data_expmaps)
            else:
                data_expmaps = _preprocess_pipeline(parsed_data)
                data = _get_standardized_rotations(data_expmaps)
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
            apply_moving_window(batches, file_data)

    conf.bvh_file_num = bvh_counter
    batches.store_effort_labels_dict()
    assert batches.batch_idx == len(batches.dict_efforts_labels.values()) - 1, f"batch_idx: {batches.batch_idx}, " \
                                                                               f"num" \
                                                                               f"labels: {len(batches.dict_efforts_labels.values())}"
    # batches.balance_similarity_classes()
    # print(f"{batches.print_len_dict_similarity_exemplars()}")


def prepare_data(rotations=True, velocities=False):
    prep_all_data_for_training(rotations=rotations, velocities=velocities)


def load_data(rotations=True, velocities=False):
    csv_file = os.path.join(conf.output_metrics_dir, f'{conf.num_task}_{conf.window_delta}.csv')
    if path.exists(conf.exemplars_dir) and not path.exists(csv_file):
        shutil.rmtree(conf.exemplars_dir)
        os.makedirs(conf.exemplars_dir)
        prepare_data(rotations=rotations, velocities=velocities)
    elif not path.exists(csv_file):
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
    # load_data(rotations=True, velocities=False)
    prepare_data()

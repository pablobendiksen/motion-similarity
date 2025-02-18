"""
static module for organizing synthetic motion data in the context of both efforts and similarity networks
"""
import sys

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

singleton_batches = Batches()


def visualize(file_bvh):
    """
    Visualize motion data from a BVH file.

    Args:
        file_bvh (str): The path to the BVH file to visualize.

    Returns:
        None
    """
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
    """
    Remove character name from a text file.

    Args:
        file (str): The path to the text file.

    Returns:
        None
    """
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


def prep_all_data_for_training(rotations=True, velocities=False, similarity_pre_processing_only=True, anim_name=None):
    """
    Prepare motion data for training.

    Args:
        rotations (bool): Whether to include rotation values in exemplars.
        velocities (bool): Whether to include velocities in exemplars.

    Returns:
        None
    """

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

    def apply_moving_window(batches, file_data):
        """
        nested function of prep_all_data_for_training()

        handles both construction of effort network batches (with rotations only, each batch has
         dim = batch_size x time_series_size x 87) and similarity network class to exemplar dict.

        Args:
            batches: instance of Batches class
            file_data: np.array comprised of preprocessed motion data + effort values + anim name

        Returns:
            None
        """
        print(f"osd::apply_moving_window(): {anim_name} ... Applying moving window to file data")
        start_index = conf.time_series_size
        end_index = file_data.shape[0]
        for i in range(start_index, end_index + conf.window_delta, conf.window_delta):
            indices = range(i - conf.time_series_size, i)
            # end of file corner case correction
            if i > end_index:
                indices = range(i - conf.time_series_size, end_index)
                exemplar = file_data[indices]
                exemplar = batches.append_to_end_file_exemplar(exemplar)
                batches.append_efforts_batch_and_labels(exemplar)
                if f == filenames[-1]:
                    batches.extend_final_batch(exemplar)
            else:
                exemplar = file_data[indices]
                batches.append_efforts_batch_and_labels(exemplar)
            if tuple_effort_list in batches.dict_similarity_exemplars.keys():
                # storing similarity class exemplar here while iterating through the file and storing effort batches
                batches.append_similarity_class_exemplar(tuple_effort_list, file_data[indices])
            if len(batches.current_batch_exemplar[batches.batch_idx]) == conf.batch_size_efforts_network:
                batches.store_efforts_batch()

    try:
        bvh_counter = 0
        bvh_frame_rate = set()
        if anim_name == "walking":
            dir_filenames = conf.bvh_files_dir_walking
            filenames = os.listdir(dir_filenames)
        elif anim_name == "pointing":
            dir_filenames = conf.bvh_files_dir_pointing
            filenames = os.listdir(dir_filenames)
        elif anim_name == "picking":
            dir_filenames = conf.bvh_files_dir_picking
            filenames = os.listdir(dir_filenames)
        else:
            raise ValueError("anim_name must be one of the following: WALKING, POINTING, PICKING")
        print(
            f"osd::prep_all_data_for_training(): {anim_name} filenames dir: {dir_filenames}, num files: {len(filenames)}")
        for f in filenames:
            if f.endswith("bvh"):
                name = path.splitext(f)[0]  # exclude extension bvh by returning the root
                name_split = name.split('_')  # get effort values from the file name
                anim = name_split[0]
                f_full_path = dir_filenames + f
                efforts_list = [float(p) for p in name.split('_')[-4:]]
                tuple_effort_list = tuple(efforts_list)
                if similarity_pre_processing_only:
                    if tuple_effort_list not in singleton_batches.dict_similarity_exemplars.keys():
                        continue
                singleton_batches.state_drive_exemplar_idx = 0
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
                    data_expmaps = data_expmaps[:, 3:]
                    data = _get_standardized_rotations(data_expmaps)
                    # data = np.hstack((data_rotations, data_expmaps))
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
                if similarity_pre_processing_only:
                    print(
                        f"Anim: {anim_name}, {bvh_counter} ... Appending similarity class exemplar for tuple: {tuple_effort_list}")
                    singleton_batches.append_similarity_class_exemplar(tuple_effort_list, file_data)
                else:
                    apply_moving_window(singleton_batches, file_data)

        conf.bvh_file_num = bvh_counter
        singleton_batches.store_effort_labels_dict()
        singleton_batches.balance_single_exemplar_similarity_classes_by_frame_count(anim_name)
        # if conf.bool_fixed_neutral_embedding:
        #     singleton_batches.pop_similarity_dict_element(key=(0, 0, 0, 0))
        # else:
        singleton_batches.move_tuple_to_similarity_dict_front(key=(0, 0, 0, 0))
        singleton_batches.convert_exemplar_np_arrays_to_tensors()
        singleton_batches.store_similarity_labels_exemplars_dict(anim_name)
        assert singleton_batches.batch_idx == len(
            singleton_batches.dict_efforts_labels.values()) - 1, f"batch_idx: {singleton_batches.batch_idx}, " \
                                                                 f"num" \
                                                                 f"labels: {len(singleton_batches.dict_efforts_labels.values())}"
        singleton_batches.verify_dict_similarity_exemplars()
    except Exception as e:
        print(f"Error in prep_all_data_for_training: {e}")
        sys.exit()


def load_data(rotations=True, velocities=False):
    """
    Load motion data for training if available, otherwise prepare it first.

    Args:
        rotations (bool): Whether to include rotations in the loaded data.
        velocities (bool): Whether to include velocities in the loaded data.

    Returns:
        partition (dict): A dictionary containing the partitioned efforts network data.
        labels_dict (dict): A dictionary containing labels (efforts values).
    """
    if not os.path.exists(conf.effort_network_exemplars_dir):
        os.makedirs(conf.effort_network_exemplars_dir)
        prep_all_data_for_training(rotations=rotations, velocities=velocities)
    # csv_file = os.path.join(conf.output_metrics_dir, f'{conf.num_task}_{conf.window_delta}.csv')
    # if path.exists(conf.effort_network_exemplars_dir) and not path.exists(csv_file):
    #     shutil.rmtree(conf.effort_network_exemplars_dir)
    #     os.makedirs(conf.effort_network_exemplars_dir)
    #     prep_all_data_for_training(rotations=rotations, velocities=velocities)
    # elif not path.exists(csv_file):
    #     prep_all_data_for_training(rotations=rotations, velocities=velocities)
    partition, labels_dict = _partition_effort_ids_and_labels()
    return partition, labels_dict


def _partition_effort_ids_and_labels(train_val_split=0.8):
    """
    Partition effort IDs and labels for training, validation, and testing.

    Args:
        train_val_split (float): The percentage of data to be used for training.

    Returns:
        partition (dict): A dictionary containing the partitioned data.
        labels_dict (dict): A dictionary containing labels for the data.
    """
    with open(conf.effort_network_exemplars_dir + conf.efforts_labels_dict_file_name, 'rb') as handle:
        labels_dict = pickle.load(handle)
    batch_ids_list = list(labels_dict.keys())
    random.shuffle(batch_ids_list)
    train_size = int(train_val_split * len(batch_ids_list))
    test_val_size = int(((1 - train_val_split) * len(batch_ids_list)) / 2)
    partition = {'train': batch_ids_list[:train_size], 'validation': batch_ids_list[train_size:test_val_size],
                 'test': batch_ids_list[-test_val_size:]}
    return partition, labels_dict


def load_similarity_data(bool_drop, anim_name, train_val_split=1.0):
    """
    Load similarity dict of all class exemplars and split across train, validation, and test sets.

    Args:
        train_val_split: float: percentage of data to be used for training versus validation and
        test sets

    Returns:
        similarity_dict: dict: partitioned similarity dict of all class exemplars
    """

    file_path = conf.similarity_exemplars_dir + anim_name + "_" + conf.similarity_dict_file_name
    if os.path.isfile(file_path):
        print(f"osd::load_similarity_data(): Generating similarity data for {anim_name} with path: {file_path}")
        prep_all_data_for_training(rotations=True, velocities=False, similarity_pre_processing_only=True,
                                   anim_name=anim_name)
    dict_similarity_classes_exemplars = pickle.load(open(file_path, "rb"))
    # Keys (sample): [(0, 0, 0, 0), (0, -1, -1, -1), (-1, 0, -1, -1), (0, 0, -1, -1), (1, 0, -1, -1)]
    # where each value is a list of a single tensor (e.g, shape: (137, 88)) and all such tensors have been made uniform
    # in their frame count
    print(f"loaded dict_similarity_classes_exemplars for anim {anim_name}")

    if bool_drop:
        conf.similarity_batch_size = 56
        dict_similarity_classes_exemplars.pop((0, 0, 0, 0))
    else:
        conf.similarity_batch_size = 57
        # ensure element of key (0, 0, 0, 0) is at the front of the dict
        singleton_batches.move_tuple_to_similarity_dict_front(key=(0, 0, 0, 0))
    # singleton_batches.dict_similarity_exemplars = dict_similarity_classes_exemplars
    # next(iter(dict_similarity_classes_exemplars.keys())) gets the first key in the dictionary
    # the length of the lone entry of the value (itself a list) somehow specifies the number of exemplars
    num_exemplars = len(dict_similarity_classes_exemplars[next(iter(dict_similarity_classes_exemplars.keys()))])
    print(f"{anim_name}: Number of total classes: {len(dict_similarity_classes_exemplars)}")
    print(f"{anim_name}: Number of total exemplars per class: {num_exemplars}")
    print(f"{anim_name}: Frame count for first exemplar: {len(dict_similarity_classes_exemplars[(0, 0, 0, 0)][0])}")
    p = np.random.permutation(num_exemplars - 1)
    train_size = int(train_val_split * num_exemplars)
    # temp change to inc val set size
    # val_and_test_size = int(((1 - train_val_split) * num_exemplars) / 2)
    val_and_test_size = int(((1 - train_val_split) * num_exemplars))
    print(f"train size: {train_size}, val and test size: {val_and_test_size}")

    train_data = {}
    validation_data = {}
    test_data = {}
    for k, v in dict_similarity_classes_exemplars.items():
        # start_index = random.randint(0, len(v) - train_size)
        # Interleave the training and validation data
        # train_data[k] = []
        # validation_data[k] = []
        # test_data[k] = []
        # for i in range(0, train_size, 2):
        #     train_data[k].append(v[i])
        #     if i < val_and_test_size:
        #         validation_data[k].append(v[i+1])
        #         test_data[k].append(v[i+1])
        train_data[k] = v[:train_size]
        if val_and_test_size == 0:
            validation_data[k] = v[:train_size]
            test_data[k] = v[:train_size]
        else:
            validation_data[k] = v[train_size:train_size + val_and_test_size]
            test_data[k] = v[train_size:train_size + val_and_test_size]

    return {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }


def balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts):
    """
    Balance the number of frames in each class exemplar to the same number of frames as the class exemplar with the
    most frames.

    Args:
        None

    Returns:
        None
    """
    print("OSD:: Balancing single exemplar similarity classes by frame count")
    return singleton_batches.balance_exemplar_similarity_classes_by_frame_count(list_similarity_dicts)

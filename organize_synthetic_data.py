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

anim_ind = {'WALKING': 0, 'POINTING': 1, 'PICKING': 2, 'WAVING': 3, 'THROWING': 4, 'AIMING': 5, 'JUMPING': 6,
            'RUNNING': 7}

parser = BVHParser()


def visualize(file_bvh):
    parsed_data = parser.parse(file_bvh)
    data_pipe = Pipeline([
        # ('param2', ConverterToRightHandedCoordinates()),
        ('param1', MocapParameterizer('expmap')),
        # ('down', DownSampler(4)),  # downsample to 30fps
        # ('stdscale', ListStandardScaler())
    ])
    data = parsed_data
    # # #
    # inv_data = data_pipe._inverse_transform([data])[0]

    # print_skel(parsed_data)
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


# all_synthetic_motions_effort (as well as Pipeline sav file) generated and saved, here
def concat_all_data_as_np(animName=None, rotations=True, velocities=False):
    column_names = None

    def _get_standardized_rotations(parsed_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_pipe_expmap = Pipeline(steps=[
                ('param', MocapParameterizer('expmap')),
                ('np', Numpyfier())
            ])
            # pickle.dump(data_pipe_expmap, open(conf.synthetic_data_pipe_file, 'wb'))
            data_expmaps = data_pipe_expmap.fit_transform([parsed_data])[0]
            # generate z-scores for all values by means of sklearn StandardScaler (i.e., standardize!)
            data_expmaps = _z_score_generator(data_expmaps)
        return data_expmaps

    def _get_standardized_velocities(parsed_data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_pipe_positions = Pipeline(steps=[
                # gives list of pymo.data.MocapData object
                ('param', MocapParameterizer('position')),
                ('np', Numpyfier())
            ])
            # all joints (corresponding to 3 columns each [Z, X,Y dimensions]) now have absolute positions
            data_velocities = data_pipe_positions.fit_transform([parsed_data])[0]
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
        # print(f"Parsed data structure columns:\n{parsed_data.values.columns}")
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

    frames = []
    file_idx_counter = 0  # unique for each file
    labels_dict = {}  # populated across all files
    # dir = conf.synthetic_data_folder
    dir = "data/effort_tmp/"
    # f represents an element from within the directory
    bvh_counter = 0
    bvh_removal_counter = 0
    bvh_frame_rate = set()
    for f in os.listdir(dir):
        if f.endswith("bvh"):
            name = path.splitext(f)[0]  # exclude extension bvh by returning the root
            # get personality/effort values from the file name
            name_split = name.split('_')
            anim = name_split[0]
            anim_extended = anim + name_split[1]
            f_full_path = dir + f
            if animName is None or str.upper(animName) == str.upper(anim):
                # extract efforts
                efforts_list = [float(p) for p in name.split('_')[-4:]]
                clear_file(f_full_path)  # remove the : from the file
                # parsed file of type pymo.data.MocapData
                parsed_data = parser.parse(f_full_path)
                if column_names is None:
                    column_names = parsed_data.values.columns.tolist()
                    column_names.insert(0, "anim")
                    column_names.insert(0, "effort_4")
                    column_names.insert(0, "effort_3")
                    column_names.insert(0, "effort_2")
                    column_names.insert(0, "effort_1")
                    print(column_names)
                # print(f"Parsed data structure columns:\n{parsed_data.values.columns}")
                bvh_frame_rate.add(parsed_data.framerate)
                if len(bvh_frame_rate) > 1:
                    fr = bvh_frame_rate.pop()
                    print(f"frame rate of: {fr} found for bvh file index {bvh_counter}.\nfile discarded")
                    continue
                    # ensure consistent frame rate across motion files (otherwise velocities miscalculated)
                assert len(bvh_frame_rate) == 1, f"More than one frame rate present!!! {bvh_frame_rate}"
                # will need to manually extend global column_names variables if velocities included
                if rotations and velocities:
                    file_name = 'data/all_synthetic_motions_velocities_effort.csv'
                    data_expmaps = _get_standardized_rotations(parsed_data)
                    # uncomment only to view corr_matrix of file
                    # create_corr_matrix(data_expmaps, name)
                    # 1st 3 columns: 'Hips_Xposition', 'Hips_Yposition', 'Hips_Zposition'; drop these
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

                # construct column array of extracted efforts themselves repeated
                # (# of frames for corresponding bvh file, 1) times
                # will be incorporated as columns to the data
                # grab efforts associated with single bvh file
                print(
                    f"parsed bvh file {bvh_counter}: {str.upper(anim_extended)} + {efforts_list}; frame count: {data.shape[0]}")
                bvh_counter += 1
                if data.shape[0] < conf.time_series_size:
                    bvh_removal_counter += 1
                f_rep = np.tile(efforts_list, (data.shape[0], 1))
                # append anim as an additional column
                a_rep = np.tile(anim_ind[str.upper(anim)], (data.shape[0], 1))
                # the animation name index will ultimately end up as the fifth column
                file_data = np.concatenate((a_rep, data), axis=1)
                # append efforts (the first 4 column(s) will be the efforts, i.e., the ML label)
                file_data = np.concatenate((f_rep, file_data), axis=1)
                file_idx_counter, labels_dict = _apply_moving_window(file_data, anim, file_idx_counter, labels_dict)
                # frames.append(file_data)

    # bvh mean frame #: 67.16410256410256, mode: 58, std_dev: 59.91612473867754
    # print(f"processed {bvh_counter} files, eliminated {bvh_removal_counter} files, using {bvh_counter - bvh_removal_counter} files")
    # motions = np.concatenate(frames)

    # all_synthetic_motions_effort.csv accessed here to generate corresponding named file that includes anim name
    # if animName:
    #     name = path.splitext(file_name)[0]  # exclude extension csv
    #     file_name = name + "_" + str.upper(animName) +'.csv'
    # pd.DataFrame(motions, columns=column_names).to_csv(file_name + "df", columns=column_names, index=False,
    #                                                    header=True,
    #                                                    sep=",")
    # we use an auxiliary file to bypass reading and transforming the data
    # np.savetxt(file_name, motions, delimiter=',')
    with open(conf.exemplars_folder + '/labels_dict.pickle', 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(len(labels_dict.keys()))

def _apply_moving_window(exemplar, anim, idx, labels_dict):
    """helper function for concat_all_data_as_np()
        exemplar: np.array
        anim: str
            animation name
        idx: int
            exemplar index
        labels_dict: dictionary
            Dictionary linking file_idx to label"""
    # exemplar = np.genfromtxt(conf.all_concatenated_motions_file, delimiter=',')
    start_index = conf.time_series_size
    end_index = exemplar.shape[0]
    for i in range(start_index, end_index+1):
        indices = range(i - conf.time_series_size, i)
        if np.all((exemplar[indices, 0:conf.feature_size + 1] == exemplar[i - conf.time_series_size,
                                                                 0:conf.feature_size + 1])):
            print(f"anim: {str(exemplar[indices[0]][conf.feature_size])}, idx: {idx}, with indices: {indices}")
            # print(exemplar[indices, 0:conf.feature_size + 1])
            labels_dict.update({idx: exemplar[indices[0]][0:conf.feature_size]})
            #drop labels and anim columns
            exemplar_tmp = np.delete(exemplar[indices], conf.feature_size + 1, axis=1)
            np.save(conf.exemplars_folder + anim + '_' + str(idx) + '.npy',
                    exemplar_tmp)
            print(f'exemplar shape: {exemplar_tmp.shape}')
            idx += 1
    return idx, labels_dict


# generate .npy for 3D array of moving window instances of size conf.time_series_size; 3D stack of 2D slices of size (time_series_size x 87)
def organize_into_time_series(rotations=True, velocities=False):
    if rotations and velocities:
        motions = np.genfromtxt(conf.all_concatenated_motions_file_2, delimiter=',')
    elif not rotations and velocities:
        motions = np.genfromtxt(conf.all_concatenated_motions_file_3, delimiter=',')
    else:
        df_motions = pd.read_csv(conf.all_concatenated_motions_file)
        motions = np.genfromtxt(conf.all_concatenated_motions_file, delimiter=',')
        # print(f"testing 1: {df_motions.shape}\n{df_motions.head(10)}")
        column_names = df_motions.columns.tolist()
        # motions = df_motions.to_numpy(copy=True)
        print(f"testing 2: {motions.shape}")
    start_index = conf.time_series_size
    end_index = motions.shape[0]

    data = []
    labels = []
    sliding_window_counter = 0
    my_bool = True
    for i in range(start_index, end_index):
        indices = range(i - conf.time_series_size, i)
        # print(df_motions.iloc[list(indices), 0:conf.feature_size + 1])

        # group dataset in chunks of size conf.time_series_size; if the first 5 rows of all elems in chunk equal to same
        # rows of its first elem (i.e., uniform) then we store this array wrt labels, then everything else
        # aka check if features and animation names are all the same for sliding_window (time_series_size); otherwise, skip
        if np.all((motions[indices, 0:conf.feature_size + 1] == motions[i - conf.time_series_size,
                                                                0:conf.feature_size + 1])):
            # if np.all((df_motions.iloc[list(indices), 0:conf.feature_size+1] == df_motions.iloc[i - conf.time_series_size,
            #                                                        0:conf.feature_size+1])):
            #     assert True is False, "EXIT"
            # if df_motions.all(df_motions.iloc[len(indices), 0:conf.feature_size + 1] == df_motions.iloc[i -
            #                                                                                         conf.time_series_size,
            #                                                         0:conf.feature_size + 1]):
            # we can now drop the fifth column (anim type) for labels, and the first 5 columns for data
            # dim: (692, 4)
            labels.append(motions[indices[0]][0:conf.feature_size])
            # dim: (692, 150, 87)
            # data.append(np.delete(motions[indices], range(0,conf.feature_size+1), axis=1))
            # drop anim column
            # if i >= 20:
            #     if my_bool:
            #         my_bool = False
            #         tmp_array = np.delete(motions[indices], conf.feature_size, axis=1)
            #         df = pd.DataFrame(tmp_array)
            #         print(df.head(10))
            #         df.to_csv('data/df_test_2.csv', index=False, header=False, sep=",")
            #         print(f"1: {tmp_array[0]}")
            #         flattened_array = tmp_array.flatten(order='C')
            #         print(f"2: {flattened_array[:10]}")
            # data.append(df_motions.iloc[indices[0], 0:conf.feature_size:])
            data.append(np.delete(motions[indices], conf.feature_size + 1, axis=1))
            sliding_window_counter += 1

    if rotations and velocities:
        np.save('data/organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy', np.array(data))
    elif not rotations and velocities:
        np.save('data/organized_synthetic_data_velocities_only_' + str(conf.time_series_size) + '.npy', np.array(data))
    else:
        # df = pd.DataFrame(data)
        # df.to_csv('data/organized_synthetic_data_' + str(conf.time_series_size), columns=column_names[6:], index=False,
        #           header=True, sep=",")
        np.save('data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy', np.array(data))
    np.save('data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy', np.array(labels))


# called when 'data/organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy' is not present
# i.e., the file with data prepped for machine learning
def prepare_data(rotations=True, velocities=False):
    if rotations and velocities:
        # enter if <all_synthetic_motions_velocities_effort.csv> file is not present; this file is needed for making the
        # 'data/organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy' file
        if not path.exists(conf.all_concatenated_motions_file_2):
            # STEP 1: Reads all the synthetic mocap data and combines them as a numpy array
            #  i.e., pre-process motion files (i.e., standardize and add velocities) and concatenate
            # creates <all_synthetic_motions_velocities_effort.csv> file
            concat_all_data_as_np(rotations=True, velocities=True)
            # STEP 2: organizes the previously concatenated motion data by subsetting it in terms of sliding window
            # creates <organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy'> file
            organize_into_time_series(rotations=True, velocities=True)
        else:
            print("path to all concatenated motions exists!")
            # STEP 2: organizes the data  with sliding window
            # creates <organized_synthetic_data_velocities_ + str(conf.time_series_size) + .npy> file
            organize_into_time_series(rotations=True, velocities=True)
    elif not rotations and velocities:
        if not path.exists(conf.all_concatenated_motions_file_3):
            concat_all_data_as_np(rotations=False, velocities=True)
            organize_into_time_series(rotations=False, velocities=True)
        else:
            print("path to all concatenated motions exists!")
            organize_into_time_series(rotations=False, velocities=True)
    else:
        if not path.exists(conf.all_concatenated_motions_file):
            concat_all_data_as_np()
            _apply_moving_window()
            # concat_all_data_as_np()
            # organize_into_time_series()
        else:
            print("path to all concatenated motions exists!")
            concat_all_data_as_np()
            # _apply_moving_window()
            # organize_into_time_series()


def load_data(rotations=True, velocities=False):
    if rotations and velocities:
        file_data = 'data/organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy'
        file_labels = 'data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy'
        if path.exists(file_data):
            data = np.load(file_data)
        else:
            prepare_data(velocities=True)
            data = np.load(file_data)
    elif not rotations and velocities:
        file_data = 'data/organized_synthetic_data_velocities_only_' + str(conf.time_series_size) + '.npy'
        file_labels = 'data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy'
        if path.exists(file_data):
            data = np.load(file_data)
        else:
            prepare_data(rotations=False, velocities=True)
            data = np.load(file_data)
    else:
        file_data = 'data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy'
        file_labels = 'data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy'
        if not path.exists(conf.exemplars_folder):
            os.makedirs(conf.exemplars_folder)
            prepare_data()

    partition, labels_dict = load_ids_and_labels()
    # labels = np.load(file_labels)
    # print(f"data shape: {data.shape}")
    return partition, labels_dict
    # return data[::conf.window_delta], labels[::conf.window_delta]

def load_ids_and_labels(train_val_split = 0.9):
    with open(conf.exemplars_folder + '/labels_dict.pickle', 'rb') as handle:
        labels_dict = pickle.load(handle)
    print(labels_dict)
    ids_list = list(labels_dict.keys())
    random.shuffle(ids_list)
    train_size = int(train_val_split * len(ids_list))
    partition = {'train': ids_list[:train_size], 'validation': ids_list[train_size:]}
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

    labels = motions[:, 0:conf.feature_size]

    indices = [i for i, val in enumerate(labels) if np.array_equal(val, efforts)]

    data = np.delete(motions[indices], range(0, conf.feature_size), axis=1)

    return data


def prepare_comparison_data():
    # concat_all_data_as_np("walking")
    # concat_all_data_as_np("picking")
    concat_all_data_as_np("pointing")


if __name__ == "__main__":
    load_data(rotations=True, velocities=False)
    # concat_all_data_as_np(rotations=True, velocities=False)
    # organize_into_time_series()
    # load_data(rotations=True, velocities=True)
    # visualize("data/cmu_motions/running_143_101.bvh")
    # to see the features assoicated with any bvh file of our training/testing data, run the following
    # dir = conf.synthetic_data_folder
    # for count, f in enumerate(os.listdir(dir)):
    #     if count == 1:
    #         f_full_path = dir + f
    #         if f.endswith("bvh"):
    #             parsed_data = parser.parse(f_full_path)
    #             print(f"Parsed data structure columns:\n{parsed_data.values.columns}")
    #             break

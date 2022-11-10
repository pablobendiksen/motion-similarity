from pymo.parsers import BVHParser

from pymo.viz_tools import *
from sklearn.preprocessing import StandardScaler
from pymo.preprocessing import *
from os import path
import conf
from sklearn.pipeline import Pipeline
import pickle
import warnings
import statistics

anim_ind = {'WALKING':0, 'POINTING': 1, 'PICKING': 2, 'WAVING':3, 'THROWING':4, 'AIMING':5}

parser = BVHParser()

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

# all_synthetic_motions_effort plus the 3 all_synthetic_motions_ANIM csv files (as well as Pipeline sav file) generated and saved, here
def concat_all_data_as_np(animName=None, velocities=False):
    frames = []
    motion_id = 0  # unique for each motion/participant
    # data/effort
    dir = conf.synthetic_data_folder
    # f represents an element from within the directory
    bvh_counter = 0
    bvh_frame_lens = []
    bvh_frame_rates = set()
    for f in os.listdir(dir):
        if f.endswith("bvh"):
            name = path.splitext(f)[0] # exclude extension bvh by returning the root
            # get personality/effort values from the file name


            anim = name.split('_')[0]
            f_full_path = dir + f
            if animName is None or str.upper(animName) == str.upper(anim):
                #extract efforts
                efforts_list = [float(p) for p in name.split('_')[1:]]
                print(f"parsing bvh file {bvh_counter}: {str.upper(anim)} + {efforts_list}")
                clear_file(f_full_path) # remove the : from the file
                # pymo.data.MocapData
                parsed_data = parser.parse(f_full_path)
                bvh_frame_rates.add(parsed_data.framerate)
                if len(bvh_frame_rates) > 1:
                    fr = bvh_frame_rates.pop()
                    print(f"frame rate of: {fr} found for bvh file index {bvh_counter}.\nfile discarded")
                    continue
                assert len(bvh_frame_rates) == 1, f"More than one frame rate present!!! {bvh_frame_rates}"
                if velocities:
                    file_name = 'data/all_synthetic_motions_velocities_effort.csv'
                    # print(f"positions pymo object column len: {len(positions[0].values.columns)}")
                    data_pipe_positions = Pipeline(steps=[
                        # gives list of pymo.data.MocapData object
                        ('param', MocapParameterizer('position')),
                        ('np', Numpyfier()),
                        # ListStandardScaler() produces  RuntimeWarning: invalid value encountered in divide
                        # normalized_track = (track - self.data_mean_) / self.data_std_ for cases whose std vector contains value(s) of 0.0
                        ('stdscale', ListStandardScaler()),
                    ])
                    data_pipe_euler = Pipeline(steps=[
                        ('param', MocapParameterizer('expmap')),
                        ('np', Numpyfier()),
                        ('stdscale', ListStandardScaler()),
                    ])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data_positions = data_pipe_positions.fit_transform([parsed_data])[0]
                        data_eulers = data_pipe_euler.fit_transform([parsed_data])[0]
                        # concat rows of data_eulers, minus first 3 (corresponds to root joint z,x,y positions)
                        # to those of data_positions, horizontally.
                        data = np.hstack((data_positions, data_eulers[:, 3:]))
                    bvh_frame_lens.append(data_positions.shape[0])
                    bvh_counter+=1
                else:
                    file_name = 'data/all_synthetic_motions_effort.csv'
                    data_pipe = Pipeline([
                        ('param', MocapParameterizer('euler')),
                        # ('rcpn', RootCentricPositionNormalizer()),
                        ('delta', RootTransformer('absolute_translation_deltas')),
                        # ('const', ConstantsRemover()),  # causes problems
                        ('np', Numpyfier()),
                        # ('down', DownSampler(2)),  # already 30fps
                        ('stdscale', ListStandardScaler())
                    ])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = data_pipe.fit_transform([parsed_data])[0]
                    pickle.dump(data_pipe, open(conf.synthetic_data_pipe_file, 'wb'))
                # construct column array of extracted efforts themselves repeated
                # (# of frames for corresponding bvh file, 1) times
                # will be incorporated as columns to the data
                # grab efforts associated with single bvh file
                f_rep = np.tile(efforts_list, (data.shape[0], 1))

                # append anim as an additional column
                a_rep = np.tile(anim_ind[str.upper(anim)], (data.shape[0], 1))
                # the animation name index will ultimately end up as the fifth column
                file_data = np.concatenate((a_rep, data), axis=1)
                # the first 4 column(s) will be the efforts
                file_data = np.concatenate((f_rep, file_data), axis=1)
                # if len(frames) > 0 and file_data.shape[1] < 64:
                #     d_rep = np.repeat(file_data[0, -1] , 64 - file_data.shape[1])
                #     file_data = np.concatenate((file_data, d_rep), axis=1)
                frames.append(file_data)

    # bvh mean frame #: 67.16410256410256, mode: 58, std_dev: 59.91612473867754
    # print(f"bvh mean frame #: {statistics.mean(bvh_frame_lens)}, mode: {statistics.mode(bvh_frame_lens)}, std_dev: {statistics.stdev(bvh_frame_lens)}")
    motions = np.concatenate(frames)
    # all_synthetic_motions_effort.csv accessed here to generate corresponding named file that includes anim name
    if animName:
        name = path.splitext(file_name)[0]  # exclude extension csv
        file_name = name + "_" + str.upper(animName) +'.csv'

    # we use an auxiliary file to bypass reading and transforming the data
    np.savetxt(file_name, motions, delimiter=',')

# generate .npy for 3D array of moving window instances of size conf.time_series_size; 3D stack of 2D slices of size (time_series_size x 87)
def organize_into_time_series(velocities=False):

    motions = np.genfromtxt(conf.all_synthetic_motions_file, delimiter=',')
    print(f"shape: {motions.shape}")
    start_index = conf.time_series_size
    end_index = motions.shape[0]

    data = []
    labels = []
    sliding_window_counter = 0
    for i in range(start_index, end_index):
        indices = range(i - conf.time_series_size, i)

        # group dataset in chunks of size conf.time_series_size; if the first 5 rows of all elems in chunk equal to same
        # rows of its first elem (i.e., uniform) then we store this array wrt labels, then everything else
        # aka check if features and animation names are all the same for sliding_window (time_series_size); otherwise, skip
        # if np.all((motions[indices] == motions[i - conf.time_series_size, 0])[:, 0]):
        if np.all((motions[indices, 0:conf.feature_size+1] == motions[i - conf.time_series_size, 0:conf.feature_size+1])):
            print(f"sliding window count: {sliding_window_counter}")
            # we can now drop the fifth column (anim type) for labels, and the first 5 columns for data
            # dim: (692, 4)
            labels.append(motions[indices[0]][0:conf.feature_size])
            # dim: (692, 150, 87)
            data.append(np.delete(motions[indices], range(0,conf.feature_size+1), axis=1))
            sliding_window_counter+=1

    if velocities:
        np.save('data/organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy', np.array(data))
    else:
        np.save('data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy', np.array(data))
    np.save('data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy', np.array(labels))


def prepare_data(velocities=False):
    # STEP 1: Reads all the synthetic mocap data and combines them as a numpy array
    if velocities:
        concat_all_data_as_np(velocities=True)
    else:
        concat_all_data_as_np()
    # STEP 2: organizes the data  with sliding windows
    organize_into_time_series()



# # Called from dcgan.py
def load_data(velocities=False):
    if velocities:
        file_data = 'data/organized_synthetic_data_velocities_' + str(conf.time_series_size) + '.npy'
        file_labels = 'data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy'
        if path.exists(file_data):
            data = np.load(file_data)
        else:
            prepare_data(velocities=True)
            data = np.load(file_data)
    else:
        file_data = 'data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy'
        file_labels = 'data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy'
        if path.exists(file_data):
            data = np.load(file_data)
        else:
            prepare_data()
            data = np.load(file_data)

    # labels are ready if data is ready
    labels = np.load(file_labels)
    # save_path = r'/Users/bendiksen/Desktop/research/virtual_humans/motion-similarity/data/'
    # df_data = pd.DataFrame(data[1])
    # df_data.to_csv(save_path + f'organized_synthetic_data_2D_slice_2_{str(conf.time_series_size)}.csv', index=None)
    # indexing data by :conf.window_delta appears to make no difference
    # data shape: 692 slices of 150x87 arrays
    # all columns contain real number values except 13-15 (values of -1, 1, 1) and 49-57 (1,0)
    print(f"data shape: {data.shape}")
    return data, labels
    # return data[::conf.window_delta], labels[::conf.window_delta]

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


    return data[0:new_data_len,:,:], labels[0:new_data_len,:,:]

# param efforts matched against synthetic_motion array indices and efforts removed from resulting array
def load_effort_animation(animName, efforts):

    name = path.splitext(conf.all_synthetic_motions_file)[0]  # exclude extension csv
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
    data = load_data(velocities=False)
    print(f"2 data shape: {data[0].shape}")

# prepare_comparison_data()

# prepare_data()
# load_data()
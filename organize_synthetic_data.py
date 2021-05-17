from pymo.parsers import BVHParser

from pymo.viz_tools import *
from pymo.preprocessing import *
from os import path
import conf
from sklearn.pipeline import Pipeline
import pickle

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


def concat_all_data_as_np(animName=None):
    frames = []
    motion_id = 0  # unique for each motion/participant
    dir = conf.synthetic_data_folder
    for f in os.listdir(dir):
        if f.endswith("bvh"):
            name = path.splitext(f)[0] # exclude extension bvh
            # get personality/effort values from the file name


            anim = name.split('_')[0]
            if animName is None or str.upper(animName) == str.upper(anim):
                feature = [float(p) for p in name.split('_')[1:]]
                # print(anim)
                # print(feature)
                f_full_path = dir + f
                clear_file(f_full_path) # remove the : from the file
                parsed_data = parser.parse(f_full_path)
                data_pipe = Pipeline([
                    ('param', MocapParameterizer('expmap')),
                    # ('rcpn', RootCentricPositionNormalizer()),
                    ('delta', RootTransformer('absolute_translation_deltas')),
                    # ('const', ConstantsRemover()),  # causes problems
                    ('np', Numpyfier()),
                    # ('down', DownSampler(2)),  # already 30fps
                    ('stdscale', ListStandardScaler())
                ])

                data = data_pipe.fit_transform([parsed_data])[0]

                pickle.dump(data_pipe, open(conf.synthetic_data_pipe_file, 'wb'))
                # append personality or effort as additional columns
                f_rep = np.tile(feature, (data.shape[0], 1))

                # append anim as an additional column
                a_rep = np.tile(anim_ind[str.upper(anim)], (data.shape[0], 1))
                # the second column is the animation name
                file_data = np.concatenate((a_rep, data), axis=1)

                # the first column(s) are the effort/personality
                file_data = np.concatenate((f_rep, file_data), axis=1)
                # if len(frames) > 0 and file_data.shape[1] < 64:
                #     d_rep = np.repeat(file_data[0, -1] , 64 - file_data.shape[1])
                #     file_data = np.concatenate((file_data, d_rep), axis=1)

                frames.append(file_data)

    all_data = np.concatenate(frames)
    file_name = conf.all_synthetic_motions_file
    if animName:
        name = path.splitext(file_name)[0]  # exclude extension csv
        file_name = name + "_" + str.upper(animName) +'.csv'

    # we use an auxiliary file to bypass reading and transforming the data
    np.savetxt(file_name, all_data, delimiter=',')


def organize_into_time_series():

    motions = np.genfromtxt(conf.all_synthetic_motions_file, delimiter=',')

    start_index = conf.time_series_size
    end_index = motions.shape[0]

    data = []
    labels = []
    for i in range(start_index, end_index):
        indices = range(i - conf.time_series_size, i)

        # check if features and animation names are all the same; otherwise, skip
        # if np.all((motions[indices] == motions[i - conf.time_series_size, 0])[:, 0]):
        if np.all((motions[indices, 0:conf.feature_size+1] == motions[i - conf.time_series_size, 0:conf.feature_size+1])):
                    # we can now drop the first column for motion id
            labels.append(motions[indices[0]][0:conf.feature_size])
            data.append(np.delete(motions[indices], range(0,conf.feature_size+1), axis=1))


    np.save('data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy', np.array(data))
    np.save('data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy', np.array(labels))


def prepare_data():
    # STEP 1: Reads all the synthetic mocap data and combines them as a numpy array
    concat_all_data_as_np()

    # STEP 2: organizes the data  with sliding windows

    organize_into_time_series()



# # Called from dcgan.py
def load_data():
    file_data = 'data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy'
    file_labels = 'data/organized_synthetic_labels_' + str(conf.time_series_size) + '.npy'
    if path.exists(file_data):
        data = np.load(file_data)
    else:
        prepare_data()
        data = np.load(file_data)

    # labels are ready if data is ready
    labels = np.load(file_labels)

    return data[::conf.window_delta], labels[::conf.window_delta]

def load_data_for_prediction():
    file = 'data/organized_synthetic_data_' + str(conf.time_series_size) + '.npy'
    if path.exists(file):
        data = np.load(file)

    else:
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


# prepare_comparison_data()

# prepare_data()
# load_data()

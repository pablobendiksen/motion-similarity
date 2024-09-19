from enum import Enum
"""
Configuration file for motion similarity project.
"""

# HYPERPARAMETERS
# number of frames per mocap exemplar (used both for effort and similarity networks)
time_series_size = 30
# number of frames to offset sliding window by between training examples
window_delta = 3
batch_size_efforts_network = 64
n_effort_epochs = 150

# complete effort network model, once trained, is saved to this file
effort_model_file = 'stored_trained_models_dir/effort_model.h5'
# complete similarity network model, once trained, is saved to this file
similarity_model_file = 'stored_trained_models_dir/similarity_model.h5'

# SIMILARITY NETWORK
# Should the neutral exemplar be dropped from the similarity network training examples
# bool_drop_neutral_exemplar = False
# # If the neutral embedding is fixed for all similarity network training examples
# bool_fixed_neutral_embedding = True
#
# squared_left_right_euc_dist = None
# squared_class_neut_dist = None

# Similarity network shape per training example; last dimension is 4 more than the
# effort network shape because effort values are included
# similarity_exemplar_dim = (time_series_size, 91)
similarity_exemplar_dim = (time_series_size, 88)
embedding_size = 32
n_similarity_epochs = 300
# Number of training examples per batch corresponds to the number of states and drives for three valued
# (poles plus zero) effort tuples
# similarity_batch_size = 56
# if not bool_fixed_neutral_embedding:
#     similarity_batch_size = 57
# else:
#     similarity_batch_size = 56

# PARALLEL PROCESSING
num_task = None
percent_files_copied = None
bvh_file_num = None
exemplar_num = None

checkpoint_root_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/model_checkpoint/"
# contains all stylized bvh files. Read by  percent_files_copy::run_percent_files_copy()
# only if a reduced subset of bvh files is desired
all_bvh_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_extended/"
# directory containing bvh files that have been stylized with 0.5 step size
# (i.e., effort values are either -1, -0.5, 0, 0.5, 1). This serves as training data for the effort predictor network
bvh_files_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_walking_105_34_552/"
effort_network_exemplars_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/effort_exemplars/"
similarity_exemplars_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/similarity_exemplars/"
output_metrics_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity" \
                     "/job_model_metrics"
efforts_labels_dict_file_name = 'labels_dict.pickle'
similarity_dict_file_name = 'similarity_labels_exemplars_dict_local.pickle'

REMOTE_MACHINE_DIR_VALUES = {
    "all_bvh_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/effort_extended/",
    "bv_files_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/effort_walking_105_34_552/",
    "exemplars_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/exemplars_dir/",
    "dict_similarity_data_path": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/exemplars_dir/",
    "output_metrics_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/job_metrics/",
    "checkpoint_root_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/model_checkpoint/"
}

# effort network generator params
# exemplar_dim changed from (time_series_size, 87) to (time_series_size, 84) as root joint absolute position removed
EFFORT_EXEMPLAR_GENERATOR_PARAMS = {'exemplar_dim': (time_series_size, 84),
                                    'batch_size': 64,
                                    'shuffle': True,
                                    'exemplars_dir': effort_network_exemplars_dir}


class BatchStrategy(Enum):
    SEMI_HARD = 1
    HARD = 2
    ALL = 3


# initialize batch strategy constant
BATCH_STRATEGY = BatchStrategy.ALL

BATCH_SEMI_HARD_PARAMS = {
    "learning_rate": 0.0001,
    "batch_size": 57,
    "num_epochs": 200,
    "embedding_size": 64
}

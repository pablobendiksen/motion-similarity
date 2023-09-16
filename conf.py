from enum import Enum

# HYPERPARAMETERS
time_series_size = 100
window_delta = 10  # int(time_series_size /
batch_size_efforts_network = 64
n_effort_epochs = 2
# n_effort_epochs = 200

# DEFAULT
effort_model_file = 'models/effort_model.h5'
buffer_size = 60000
num_efforts = 4
exemplar_dim_effort_network = (time_series_size, 87)

# SIMILARITY NETWORK
bool_fixed_neutral_embedding = True
similarity_exemplar_dim = (time_series_size, 91)
embedding_size = 30
similarity_batch_size = 56
if not bool_fixed_neutral_embedding:
    similarity_batch_size = 57
n_similarity_epochs = 500

# PARALLEL PROCESSING
num_task = None
percent_files_copied = None
bvh_file_num = None
exemplar_num = None

checkpoint_root_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/model_checkpoint/"
# currently unused as bvh_files_dir containes walking bvh files stylized with 0.5 step size
all_bvh_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_extended/"
# bvh_subsets_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/data
# /effort_tmp/"
bvh_files_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_walking_105_34_552_tmp/"
exemplars_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/tmp/"
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

BATCH_SEMI_HARD_PARAMS = {
    "learning_rate": 0.0001,
    "batch_size": 57,
    "num_epochs": 200,
    "embedding_size": 64
}


class BatchStrategy(Enum):
    SEMI_HARD = 1
    HARD = 2
    ALL = 3

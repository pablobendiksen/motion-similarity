# DEFAULT
effort_model_file = 'models/effort_model.h5'
checkpoint_dir = f'model_checkpoint_'
buffer_size = 60000
num_efforts = 4

# HYPERPARAMETERS
time_series_size = 100
window_delta = 10 # int(time_series_size /
batch_size_efforts_predictor = 64
n_epochs = 100

# PARALLEL PROCESSING
num_task = None
percent_files_copied = None
bvh_file_num = None
exemplar_num = None

all_bvh_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_extended/"
bvh_subsets_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/data/effort_tmp/"
exemplars_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/tmp/"
metrics_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/job_model_accuracies"

REMOTE_MACHINE_DIR_VALUES = {
    'all_bvh_dir': "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/effort_extended/",
    "bv_subsets_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/effort_subset/",
    "exemplars_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/exemplars_dir/",
    "metrics_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/job_model_accuracies"
}


checkpoint_path = 'checkpoints'
data_pipe_file = 'models/pipeline.sav'
synthetic_data_folder = "data/effort/"
effort_model_file = 'models/effort_model.h5'

synthetic_data_pipe_file = 'models/pipeline_synthetic.sav'
all_concatenated_motions_file = 'data/all_synthetic_motions_effort.csv'
all_concatenated_motions_file_2 = 'data/all_synthetic_motions_velocities_effort.csv'
# all_concatenated_motions_file_3 = 'data/all_synthetic_motions_velocities_only_effort.csv'
synthetic_model_file = 'models/synthetic_LMA.h5'
pdist_model_file = 'models/pdist.h5'
# all_bvh_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/data/effort_tmp/"
all_bvh_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_subset/"
# all_exemplars_folder = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity" \
#                        "/data_tmp/"
all_exemplars_folder_3 = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/data_tmp/"

n_participants = 144
time_series_size = 100
window_delta = 1 #int(time_series_size / 10)
batch_size = 32
buffer_size = 60000
shard_size = buffer_size
kernel_size = 5
n_epochs = 50
feature_size = 4




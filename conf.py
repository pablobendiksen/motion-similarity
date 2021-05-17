checkpoint_path = 'checkpoints'
data_pipe_file = 'models/pipeline.sav'

pdist_model_file = 'models/pdist.h5'


synthetic_data_folder = "data/effort/"
synthetic_data_pipe_file = 'models/pipeline_synthetic.sav'
all_synthetic_motions_file = 'data/all_synthetic_motions_effort.csv'

synthetic_model_file = 'models/synthetic_LMA.h5'
n_participants = 144
# time_series_size = 90
time_series_size = 150

window_delta = 1 #int(time_series_size / 10)
latent_dim = 100
batch_size = 32
buffer_size = 60000
kernel_size = 5
n_epochs = 50
feature_size = 4




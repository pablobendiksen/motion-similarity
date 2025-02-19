from enum import Enum
from typing import Dict, Optional
import os


class Config:
    def __init__(self, task_index: Optional[str] = None):
        # Initialize with task index if running on remote machine
        self.num_task = task_index
        self.is_remote = task_index is not None

        # HYPERPARAMETERS
        self.time_series_size = 30
        self.window_delta = 3
        self.batch_size_efforts_network = 64
        self.n_effort_epochs = 150

        # Model files
        self.effort_model_file = 'stored_trained_models_dir/effort_model.h5'
        self.similarity_model_file = 'stored_trained_models_dir/similarity_model.h5'

        # Network dimensions
        self.similarity_exemplar_dim = (137, 88)
        self.embedding_size = 32
        self.n_similarity_epochs = 500
        self.similarity_batch_size = 57
        self.similarity_dict_file_name = 'similarity_labels_exemplars_dict_local.pickle'

        # File paths - initialized with local paths by default
        self._base_local_paths = {
            "checkpoint_root_dir": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/model_checkpoint/",
            "all_bvh_dir": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_extended/",
            "bvh_files_dir_walking": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/walking_perform_user_study_1/",
            "bvh_files_dir_pointing": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/pointing_perform_user_study_1/",
            "bvh_files_dir_picking": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/picking_perform_user_study_1/",
            "effort_network_exemplars_dir": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/effort_exemplars/",
            "similarity_exemplars_dir": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/similarity_exemplars/",
            "output_metrics_dir": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/job_model_metrics"
        }

        self._remote_machine_paths = {
            "checkpoint_root_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/model_checkpoint/",
            "all_bvh_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/effort_extended/",
            "bvh_files_dir_walking": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/walking_perform_user_study_1/",
            "bvh_files_dir_pointing": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/pointing_perform_user_study_1/",
            "bvh_files_dir_picking": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/picking_perform_user_study_1/",
            "effort_network_exemplars_dir": "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/effort_exemplars/",
            "similarity_exemplars_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/exemplars_dir/similarity_exemplars/",
            "output_metrics_dir": "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/motion-similarity/job_metrics/"
        }

        # Set paths based on environment
        self._set_paths()

        # Dictionary file names
        self.efforts_labels_dict_file_name = 'labels_dict.pickle'
        self.similarity_dict_file_name = 'similarity_labels_exemplars_dict_local.pickle'

        # Effort network generator parameters
        self.EFFORT_EXEMPLAR_GENERATOR_PARAMS = {
            'exemplar_dim': (self.time_series_size, 84),
            'batch_size': 64,
            'shuffle': True,
            'exemplars_dir': self.effort_network_exemplars_dir
        }

    def _set_paths(self) -> None:
        """Set paths based on whether we're running locally or remotely"""
        if self.is_remote:
            # Update paths for remote execution
            self.checkpoint_root_dir = os.path.join(self._remote_machine_paths['checkpoint_root_dir'],
                                                    f"{self.num_task}/")
            self.all_bvh_dir = self._remote_machine_paths['all_bvh_dir']
            self.bvh_files_dir_walking = self._remote_machine_paths['bvh_files_dir_walking']
            self.bvh_files_dir_pointing = self._remote_machine_paths['bvh_files_dir_pointing']
            self.bvh_files_dir_picking = self._remote_machine_paths['bvh_files_dir_picking']
            self.effort_network_exemplars_dir = self._remote_machine_paths['effort_network_exemplars_dir']
            self.output_metrics_dir = self._remote_machine_paths['output_metrics_dir']
        else:
            # Use local paths
            self.__dict__.update(self._base_local_paths)

    def ensure_directories_exist(self) -> None:
        """Ensure all required directories exist"""
        directories = [self.output_metrics_dir, self.checkpoint_root_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")


class BatchStrategy(Enum):
    SEMI_HARD = 1
    HARD = 2
    ALL = 3


# Initialize configuration
config = Config()

# Batch strategy configuration
BATCH_STRATEGY = BatchStrategy.HARD

BATCH_SEMI_HARD_PARAMS = {
    "learning_rate": 0.0001,
    "batch_size": 57,
    "num_epochs": 200,
    "embedding_size": 64
}
"""
Program entry point. Loads data for, trains, and evaluates both effort and similarity networks.

If command line argument is provided, program assumes value represents a parallel job index and treats the
execution as a remote machine run. Otherwise, the program is assumed to be running locally.
"""

import os
import sys
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(curr_path + '\networks')
from networks.similarity_network import SimilarityNetwork
from networks.similarity_data_loader import SimilarityDataLoader
from networks.triplet_mining import TripletMining
from Config import Config
import src.organize_synthetic_data as osd
import tensorflow as tf


def check_gpu_access():
    # Check if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ TensorFlow detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  - {gpu}")

        # Set TensorFlow to use GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Run a small computation on the GPU
        with tf.device('/GPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b
        print(f"✅ GPU computation successful: {c.numpy()}")
    else:
        print("❌ No GPU detected by TensorFlow.")


if __name__ == '__main__':
    check_gpu_access()

    # Initialize configuration with task index if provided (happens only for remote machine)
    task_index = sys.argv[1] if len(sys.argv) > 1 else None
    config = Config(task_index)

    # Ensure required directories exist
    config.ensure_directories_exist()

    # Architecture variant is either from task index or default
    arch_variant = int(config.num_task) if config.num_task else 0

    # load effort data and train effort network
    # batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
    # effort_train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict,
    #                                              **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
    # effort_validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict,
    #                                                   **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
    # effort_test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict,
    #                                             **conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS)
    # effort_network = EffortNetwork(train_generator=effort_train_generator,
    #                                validation_generator=effort_validation_generator,
    #                                test_generator=effort_test_generator, checkpoint_dir=checkpoint_dir)
    # start_time = time.time()
    # history = effort_network.run_model_training()
    # minutes_tot_time = (time.time() - start_time) / 60
    # effort_network.write_out_training_results(minutes_tot_time)
    # collect_job_metrics.collect_job_metrics()

    bool_drop_neutral_exemplar = False
    bool_fixed_neutral_embedding = False
    squared_left_right_euc_dist = True
    squared_class_neut_euc_dist = False

    # load similarity data and train similarity network
    walking_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "walking", config)
    pointing_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing", config)
    picking_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking", config)
    list_similarity_dicts = [walking_similarity_dict_partition["train"], pointing_similarity_dict_partition["train"],
                             picking_similarity_dict_partition["train"]]
    list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts)
    walking_triplet_mining = TripletMining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist, "walking", config)
    pointing_triplet_mining = TripletMining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist, "pointing", config)
    picking_triplet_mining = TripletMining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist, "picking", config)

    # similarity_train_loader = SimilarityDataLoader(walking_similarity_dict_partition['train'])
    # similarity_validation_loader = SimilarityDataLoader(walking_similarity_dict_partition['validation'])
    # similarity_test_loader = SimilarityDataLoader(walking_similarity_dict_partition['test'])

    similarity_train_loader = SimilarityDataLoader(list_similarity_dicts)
    similarity_validation_loader = SimilarityDataLoader(list_similarity_dicts)
    similarity_test_loader = SimilarityDataLoader(list_similarity_dicts)

    similarity_network = SimilarityNetwork(train_loader=similarity_train_loader,
                                           validation_loader=similarity_validation_loader,
                                           test_loader=similarity_test_loader,
                                           checkpoint_root_dir=config.checkpoint_root_dir,
                                           triplet_modules=[walking_triplet_mining, pointing_triplet_mining,
                                                            picking_triplet_mining],
                                           architecture_variant=arch_variant)
    similarity_network.run_model_training()

    # similarity_network.evaluate()
    print(f"--------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------")

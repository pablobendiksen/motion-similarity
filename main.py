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
import src.organize_synthetic_data as osd
import conf

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("running on remote machine")
        conf.num_task = sys.argv[1]  # This is the Slurm job array index
        arch_variant = int(conf.num_task)  # Convert it to an integer
    else:
        conf.num_task = None
        arch_variant = 0  # Default architecture variant

    if conf.num_task:
        conf.all_bvh_dir = conf.REMOTE_MACHINE_DIR_VALUES['all_bvh_dir']
        conf.bvh_files_dir_walking = conf.REMOTE_MACHINE_DIR_VALUES['bv_files_dir']
        conf.effort_network_exemplars_dir = conf.EFFORT_EXEMPLAR_GENERATOR_PARAMS['exemplars_dir'] = (conf.REMOTE_MACHINE_DIR_VALUES['exemplars_dir'] +
                                                                                                      conf.num_task + '/')
        conf.output_metrics_dir = conf.REMOTE_MACHINE_DIR_VALUES['output_metrics_dir']
        conf.checkpoint_root_dir = conf.REMOTE_MACHINE_DIR_VALUES['checkpoint_root_dir'] + conf.num_task + '/'

    if not os.path.exists(conf.output_metrics_dir):
        os.makedirs(conf.output_metrics_dir)

    if not os.path.exists(conf.checkpoint_root_dir):
        os.makedirs(conf.checkpoint_root_dir)
        print(f"created new similarity network checkpoint directory: {conf.checkpoint_root_dir}")

    # load effort data and train effort network
    batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
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
    walking_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "walking")
    pointing_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "pointing")
    picking_similarity_dict_partition = osd.load_similarity_data(bool_drop_neutral_exemplar, "picking")
    list_similarity_dicts = [walking_similarity_dict_partition["train"], pointing_similarity_dict_partition["train"],
                             picking_similarity_dict_partition["train"]]
    list_similarity_dicts = osd.balance_single_exemplar_similarity_classes_by_frame_count(list_similarity_dicts)
    walking_triplet_mining = TripletMining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist, "walking")
    pointing_triplet_mining = TripletMining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist, "pointing")
    picking_triplet_mining = TripletMining(bool_drop_neutral_exemplar, bool_fixed_neutral_embedding, squared_left_right_euc_dist, squared_class_neut_euc_dist, "picking")

    # similarity_train_loader = SimilarityDataLoader(walking_similarity_dict_partition['train'])
    # similarity_validation_loader = SimilarityDataLoader(walking_similarity_dict_partition['validation'])
    # similarity_test_loader = SimilarityDataLoader(walking_similarity_dict_partition['test'])

    similarity_train_loader = SimilarityDataLoader(list_similarity_dicts)
    similarity_validation_loader = SimilarityDataLoader(list_similarity_dicts)
    similarity_test_loader = SimilarityDataLoader(list_similarity_dicts)

    similarity_network = SimilarityNetwork(train_loader=similarity_train_loader,
                                           validation_loader=similarity_validation_loader,
                                           test_loader=similarity_test_loader,
                                           checkpoint_root_dir=conf.checkpoint_root_dir,
                                           triplet_modules=[walking_triplet_mining, pointing_triplet_mining,
                                                            picking_triplet_mining],
                                           architecture_variant=arch_variant)
    similarity_network.run_model_training()

    # similarity_network.evaluate()
    print(f"--------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------")
    print(f"--------------------------------------------------------------------------")

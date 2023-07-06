import shutil
import os
import sys
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.append(curr_path + '\..\networks')
from networks.effort_network import EffortNetwork
from networks.generator import MotionDataGenerator
import organize_synthetic_data as osd
import aux.percent_files_copy as percent_files_copy
import time
import conf


def delete_exemplars_dir(task_num):
    directory = conf.exemplars_dir + task_num + '/'
    if os.path.exists(directory):
        if not os.listdir(directory):
            print(f"Directory is empty: {directory}")
        else:
            shutil.rmtree(directory)
            print(f"Deleted directory: {directory}")
    else:
        print(f"Directory does not exist: {directory}")


remote_sliding_window_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

# effort network generator params
params = {'exemplar_dim': (100, 87),
          'batch_size': conf.batch_size_efforts_network,
          'shuffle': True,
          'exemplars_dir': conf.exemplars_dir}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        conf.num_task = sys.argv[1]
    else:
        conf.num_task = None
    print(f"task number: {conf.num_task}")
    sliding_window_sizes = remote_sliding_window_sizes
    sliding_window_sizes = [10]
    if conf.num_task:
        conf.all_bvh_dir = conf.REMOTE_MACHINE_DIR_VALUES['all_bvh_dir']
        conf.bvh_files_dir = conf.REMOTE_MACHINE_DIR_VALUES['bv_subsets_dir']
        conf.exemplars_dir = params['exemplars_dir'] = conf.REMOTE_MACHINE_DIR_VALUES['exemplars_dir'] + \
            conf.num_task + '/'
        conf.output_metrics_dir = conf.REMOTE_MACHINE_DIR_VALUES['output_metrics_dir']
        conf.checkpoint_root_dir = conf.REMOTE_MACHINE_DIR_VALUES['checkpoint_root_dir'] + conf.num_task + '/'
        sliding_window_sizes = remote_sliding_window_sizes

    if not os.path.exists(conf.checkpoint_root_dir):
        os.makedirs(conf.checkpoint_root_dir)

    for window_size in sliding_window_sizes:
        checkpoint_dir = '_'.join(filter(None, [conf.checkpoint_root_dir, str(window_size)]))
        conf.window_delta = window_size
        batch_ids_partition, labels_dict = osd.load_data(rotations=True, velocities=False)
        print(f"number of batches: {len(labels_dict.keys())}")
        print(f"type: {type(labels_dict[1])}")
        train_generator = MotionDataGenerator(batch_ids_partition['train'], labels_dict, **params)
        validation_generator = MotionDataGenerator(batch_ids_partition['validation'], labels_dict, **params)
        test_generator = MotionDataGenerator(batch_ids_partition['test'], labels_dict, **params)
        effort_network = EffortNetwork(exemplar_shape=params["exemplar_dim"], two_d_conv=False, model_num=1)
        start_time = time.time()
        history = effort_network.run_model_training(train_generator, validation_generator,
                                                    checkpoint_dir)
        tot_time = (time.time() - start_time) / 60
        index_window_size = conf.num_task + '.' + str(window_size)
        effort_network.write_out_eval_accuracy(test_generator, conf.num_task, checkpoint_dir, tot_time)

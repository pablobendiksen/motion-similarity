import os
import conf
import random
import shutil

# Copy a percentage of input directory files to output directory

INPUT_DIR = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_extended"
OUTPUT_DIR = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_subset"
PERCENT_FILES_TO_COPY = 12.5
FILE_COUNT = 0
NUM_SAMPLES = None

def rename_blender_to_unity_files(dir):
    for f in os.listdir(dir):
        if "|" in f:
            sub_strings = f.split("|")
            os.rename(dir + f, dir + sub_strings[1])

def run_percent_files_copy(task_num):
    # stylized jumping + polarized stylized files: count: 14838
    # num samps at % 6.25: 911
    # after window application num_samps: data shape: (81200, 100, 87)
    # num samps at % 12.5: 1830
    # after window application num_samps: data shape: (160454, 100, 87)
    percent_copy = 6.0 + int(task_num)*2
    conf.percent_files_copied = percent_copy
    INPUT_DIR = conf.all_bvh_dir
    OUTPUT_DIR = conf.bvh_subsets_dir + task_num
    file_count = 0
    for _, _ in enumerate(os.listdir(INPUT_DIR)):
        file_count += 1
    # print(f"count: {file_count}")
    NUM_SAMPLES = int(percent_copy / 100 * file_count)
    # print(f"num samps: {NUM_SAMPLES}")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print(f"created new directory: {OUTPUT_DIR}")

    filenames = os.listdir(INPUT_DIR)
    file_subset_names = []
    # for k, path in enumerate(os.listdir(INPUT_DIR)):
    #     if k < NUM_SAMPLES:
    #         file_subset_names.append(str(path))  # because path is object not string
    #     else:
    #         i = random.randint(0, k)
    #         if i < NUM_SAMPLES:
    #             file_subset_names[i] = str(path)
    # print(f"file_subset_names {file_subset_names}")
    num_files = len(filenames)
    for i in NUM_SAMPLES:
        # j = random.randint(0, num_files)
        # file_subset_names.append(filenames[j])
        file_subset_names.append(filenames[i])



    for file_name in file_subset_names:
        print(f"copying: {file_name}")
        shutil.copy(os.path.join(INPUT_DIR, file_name), os.path.join(OUTPUT_DIR))





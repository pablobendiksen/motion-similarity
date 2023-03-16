import os
from pathlib import Path
import random
import shutil

# Copy a percentage of input directory files to output directory

INPUT_DIR = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_extended"
OUTPUT_DIR = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_subset"
PERCENT_FILES_TO_COPY = 12.5
FILE_COUNT = 0
NUM_SAMPLES = None

if __name__ == '__main__':
    # stylized jumping + polarized stylized files: count: 14838
    # num samps at % 6.25: 911
    # after window application num_samps: data shape: (81200, 100, 87)
    # num samps at % 12.5: 1830
    # after window application num_samps: data shape: (160454, 100, 87)
    for _, _ in enumerate(os.listdir(INPUT_DIR)):
        FILE_COUNT+=1
    print(f"count: {FILE_COUNT}")
    NUM_SAMPLES = int(PERCENT_FILES_TO_COPY / 100 * FILE_COUNT)
    print(f"num samps: {NUM_SAMPLES}")


    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        print(f"created new directory: {OUTPUT_DIR}")

    file_name_list = []
    for k, path in enumerate(os.listdir(INPUT_DIR)):
        if k < NUM_SAMPLES:
            file_name_list.append(str(path))  # because path is object not string
        else:
            i = random.randint(0, k)
            if i < NUM_SAMPLES:
                file_name_list[i] = str(path)
    print(f"file_name_list {file_name_list}")

    for file_name in file_name_list:
        print(f"copying: {file_name}")
        shutil.copy(os.path.join(INPUT_DIR, file_name), os.path.join(OUTPUT_DIR))





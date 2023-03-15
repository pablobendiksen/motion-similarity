import os
import conf

bvh_extended_dir = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/effort_subset/"

def rename_blender_to_unity_files(dir):
    for f in os.listdir(dir):
        if "|" in f:
            sub_strings = f.split("|")
            os.rename(bvh_extended_dir + f, bvh_extended_dir + sub_strings[1])

if __name__ == "__main__":
    rename_blender_to_unity_files(conf.all_bvh_dir)
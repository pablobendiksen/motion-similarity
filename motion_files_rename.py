import os
import conf

effort_extended_dir = "data/effort_extended/"

def rename_blender_to_unity_files(dir):
    for f in os.listdir(dir):
        if "|" in f:
            sub_strings = f.split("|")
            sub_strings[0] += sub_strings[1]
            os.rename(effort_extended_dir + f, effort_extended_dir+ sub_strings[0])

if __name__ == "__main__":
    rename_blender_to_unity_files(effort_extended_dir)
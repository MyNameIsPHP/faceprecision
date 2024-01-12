# Promt:

    # User
    # 2 have 2 datasets:
    # Each dataset have 3 folders: Train, Test and Val. Each folder contains subfolders that contain images
    # Write Python code to merge 2 datasets into a big one

# instead of merge 2 folders, i want to merge a list of folder

import os
import shutil

def merge_datasets(src_dirs, dst):
    def copy_and_rename(src_dir, dst_dir, identifier):
        # Counter for unique file names
        count = 1
        for file in os.listdir(src_dir):
            file_ext = os.path.splitext(file)[1]  # Extract file extension
            new_file_name = f"{identifier}_{count}{file_ext}"
            shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, new_file_name))
            count += 1

    # Creating main directories in the new dataset
    for main_dir in ['Train', 'Test', 'Val']:
        os.makedirs(os.path.join(dst, main_dir), exist_ok=True)

        # Merging subdirectories from all source datasets
        for src_dir in src_dirs:
            src_main_dir = os.path.join(src_dir, main_dir)
            for subdir in os.listdir(src_main_dir):
                # Create subdirectory in the destination if it doesn't exist
                dest_subdir = os.path.join(dst, main_dir, subdir)
                os.makedirs(dest_subdir, exist_ok=True)

                # Copy and rename files from source subdirectory to destination subdirectory
                src_subdir = os.path.join(src_main_dir, subdir)
                # Extract an identifier from the source directory name (e.g., 'ds1', 'ds2')
                identifier = os.path.basename(src_dir)
                copy_and_rename(src_subdir, dest_subdir, identifier)

# List of source dataset directories to merge
src_datasets = ['emotion_data_samp_1', 'emotion_data_samp_2', 'fer2013']
dst_dataset = 'emotion_all'  # Destination dataset directory

# Merge the datasets
merge_datasets(src_datasets, dst_dataset)

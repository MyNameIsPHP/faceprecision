import os
import shutil

# Define the paths to the datasets and the merged dataset
datasets = ["emotion_data_main", "fer2013"]
merged_dataset_path = "merged"

# Define a list of folders to merge
folders_to_merge = [f.path.split('/')[1] for f in os.scandir(datasets[0]) if f.is_dir()]
print(folders_to_merge)
# Function to merge a list of datasets and folders
def merge_datasets(datasets, merged_dataset_path, folders_to_merge):
    # Create the merged dataset directory if it doesn't exist
    if not os.path.exists(merged_dataset_path):
        os.makedirs(merged_dataset_path)

    for folder in folders_to_merge:
        merged_folder_path = os.path.join(merged_dataset_path, folder)

        # Create the merged folder if it doesn't exist
        if not os.path.exists(merged_folder_path):
            os.makedirs(merged_folder_path)

        # Function to copy files with unique names
        def copy_files_with_unique_names(src_folder, dst_folder):
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(dst_folder, file)
                    
                    # Check if the file already exists in the destination folder
                    if os.path.exists(dst_file):
                        # Generate a unique filename by adding a suffix
                        base_name, ext = os.path.splitext(file)
                        i = 1
                        while os.path.exists(dst_file):
                            new_file_name = f"{base_name}_{i}{ext}"
                            dst_file = os.path.join(dst_folder, new_file_name)
                            i += 1

                    shutil.copy(src_file, dst_file)

        # Loop through the datasets and copy the contents to the merged folder
        for dataset_path in datasets:
            dataset_folder_path = os.path.join(dataset_path, folder)
            copy_files_with_unique_names(dataset_folder_path, merged_folder_path)

# Merge the specified folders from the datasets
merge_datasets(datasets, merged_dataset_path, folders_to_merge)

print("Folders merged successfully!")
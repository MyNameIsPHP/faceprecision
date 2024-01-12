# Prompt: I have a "train" folder for image classification task that contains subfolders, each subfolder contains images belongs to a class (name of folder is name of the class), write Python code to randomly divide the dataset folder into "Train", "Val"

import os
import shutil
import random

def split_data(source_folder, train_folder, val_folder, split_size):
    """
    Splits the dataset into training and validation sets.

    Args:
    source_folder (str): Path to the source folder containing class subfolders.
    train_folder (str): Path to the folder where training set subfolders will be created.
    val_folder (str): Path to the folder where validation set subfolders will be created.
    split_size (float): The fraction of data to be used for training.
    """
    classes = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    
    for cls in classes:
        cls_source = os.path.join(source_folder, cls)
        cls_train = os.path.join(train_folder, cls)
        cls_val = os.path.join(val_folder, cls)

        if not os.path.exists(cls_train):
            os.makedirs(cls_train)
        if not os.path.exists(cls_val):
            os.makedirs(cls_val)

        images = os.listdir(cls_source)
        random.shuffle(images)
        split_point = int(split_size * len(images))

        for img in images[:split_point]:
            shutil.copy(os.path.join(cls_source, img), cls_train)
        for img in images[split_point:]:
            shutil.copy(os.path.join(cls_source, img), cls_val)

# Example usage
split_data('fer2013/train', 'fer2013/Train', 'fer2013/Val', 0.8)
# Prompt:
# I have a "dataset" folder for image classification task that contains subfolders, each subfolder contains images belongs to a class (name of folder is name of the class), write Python code to randomly divide the dataset folder into "Train", "Test" and "Val" 

import os
import shutil
import random

def split_dataset(dataset_path, save_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        images = [img for img in os.listdir(cls_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        random.shuffle(images)

        train_size = int(train_ratio * len(images))
        val_size = int(val_ratio * len(images))

        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        for img in train_images:
            target_save_path =  os.path.join(save_path, 'Train', cls)
            if not os.path.exists(target_save_path):
                os.makedirs(target_save_path)
            print(os.path.join(cls_path, img), "->", os.path.join(target_save_path, img))
            shutil.copy(os.path.join(cls_path, img), os.path.join(target_save_path, img))

        for img in val_images:
            target_save_path =  os.path.join(save_path, 'Val', cls)
            if not os.path.exists(target_save_path):
                os.makedirs(target_save_path)
            print(os.path.join(cls_path, img), "->", os.path.join(target_save_path, img))
            shutil.copy(os.path.join(cls_path, img), os.path.join(target_save_path, img))

        for img in test_images:
            target_save_path =  os.path.join(save_path, 'Test', cls)
            if not os.path.exists(target_save_path):
                os.makedirs(target_save_path)
            print(os.path.join(cls_path, img), "->", os.path.join(target_save_path, img))
            shutil.copy(os.path.join(cls_path, img), os.path.join(target_save_path, img))

# Example usage
dataset_path = 'non_face_detector'  # Replace with your dataset path
save_path = 'non_face_detector_samp_1'

split_dataset(dataset_path, save_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

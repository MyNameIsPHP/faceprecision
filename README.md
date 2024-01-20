# Faceprecision 
Faceprecision is a comprehensive face analysis project leveraging advanced deep learning and computer vision techniques. This project includes a modules and source codes for detecting and analyzing various facial attributes such as [age](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/age_gender_classification), [emotion](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/emotion_detection), [gender](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/age_gender_classification), [the presence of a mask](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/mask_detection), [race](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/race_skintone_classification) and [skin tone](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/race_skintone_classification).

## Methods
The implemented facial recognition system employs YOLOv8 for face detection due to its superior speed, accuracy, and low latency. To enhance the system's capabilities, the approach is compartmentalized into five categories: Age and Gender, Emotion, Mask, Race, and Skintone. The chosen methods and models aim to achieve a balance between accuracy and efficiency, making the system suitable for real-time image processing and facial recognition applications, you can see more details at:

- [Age and Gender Detection (Transfer Learning and Multitask Learning)](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/age_gender_classification)
- [Emotion Detection (CNN with Attention)](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/emotion_detection)
- [Mask Detection (CNN with Attention)](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/mask_detection)
- [Race Detection (Ensemble Learning)](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/race_skintone_classification)
- [Skintone Detection](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/race_skintone_classification)


## Datasets
In supplement to the primary dataset, we integrated the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset to enhance Age and Gender identification, Race identification and [MST-e](https://skintone.google/mste-dataset) dataset to enhance Skintone identification. Moreover, to enrich the training dataset, we implemented basic augmentation techniques. This strategy aims to introduce greater diversity into the dataset, thereby improving the model's robustness and performance.

## Repository Structure

The faceprecision repository is organized as follows:

- Python Files:
    - [evaluate.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/evaluate.py): Script for evaluating face analysis models.
    - [process_folder.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/process_folder.py): Script for processing a folder of images for face analysis.
    - [webcam.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/webcam.py): Script for real-time face analysis using a webcam.
- Folders:
    - [face_analysis](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis): Contains subfolders for each face analysis aspect ([age_gender_classification](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/age_gender_classification), [emotion_detection](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/emotion_detection), [mask_detection](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/mask_detection), [race_skintone_classification](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis/race_skintone_classification)), each with models and weights folders.
    - [face_detection](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_detection): Includes yolov8 and non_face_classification subfolders, each with models and weights folders.
- Models and Weights:
    - models folder: Contains files, Google Colab links for training and evaluating models, and files for converting H5 to ONNX format.
    - weights folder: Contains weights of models in ONNX format.
        
        
## Features
- Face Analysis: Leverage advanced models for detecting age, emotion, gender, mask usage, and race/skintone.
- Face Detection: Utilize state-of-the-art YOLOv8 and non-face classification models for accurate face detection.
- Real-Time Analysis: Perform face analysis in real-time using the [webcam.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/webcam.py) script.
- Folder Processing: Efficiently process an entire folder of images for comprehensive face analysis.

## Usage

Detailed usage instructions for each script ([evaluate.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/evaluate.py), [process_folder.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/process_folder.py), [webcam.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/webcam.py)) are provided in their respective comments and documentation. Refer to these for specific command-line arguments and options.



## Installation & Open Webcam
Ensure you have Anaconda or Miniconda installed. Follow these steps to set up the environment:

- Create a Conda Environment:
    ```shell
    conda create -n faceprecision python=3.8
    conda activate faceprecision
    ```

- Clone the Repository:
    ```shell
    git clone https://github.com/MyNameIsPHP/faceprecision.git
    cd faceprecision
    ```

- Install Requirements:
    ```shell
    pip install -r requirements.txt
    ```
- Run command to open your webcam:
    ```shell
    python3 webcam.py
    ```
To successfully run the Python scripts ([evaluate.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/evaluate.py), [process_folder.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/process_folder.py) and [webcam.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/webcam.py)) in the "faceprecision" repository, certain files and folders must be present and correctly configured. Here's a breakdown of the required components for each script:
1. [evaluate.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/evaluate.py)

    - Data Folder: Contains the dataset on which the evaluation will be performed. The path to this folder should be correctly specified in the script or passed as an argument.
    - Models and Weights: The models and their corresponding weights used for face analysis must be available. These are likely located in the [face_analysis/](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis) and [face_detection/](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_detection) folders, specifically in their respective models and weights subfolders.
    - Output Folder: A folder to store the evaluation results. The script may generate output files or reports in this folder.

2. [process_folder.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/process_folder.py)

    - Input Data Folder: The folder containing images to be processed. This path should be specified in the script or provided as an argument.
    - Models and Weights: Similar to [evaluate.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/evaluate.py), this script requires the models and weights for performing face analysis, located in the [face_analysis/](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_analysis) and [face_detection/](https://github.com/MyNameIsPHP/faceprecision/tree/main/faceprecision/face_detection) folders.
    - Output Folder: A folder where the processed results, such as annotated images or extracted data, will be saved.

3. [webcam.py](https://github.com/MyNameIsPHP/faceprecision/blob/main/faceprecision/webcam.py)

    - Models and Weights: For real-time analysis using webcam input, this script requires access to the face analysis and detection models and their weights, stored in the face_analysis/ and face_detection/ folders.
    - Webcam Access: Ensure that the device running the script has a functioning webcam and necessary permissions are granted for the script to access it.

## Contributing

We welcome contributions! Please submit pull requests for improvements, bug fixes, or enhancements.

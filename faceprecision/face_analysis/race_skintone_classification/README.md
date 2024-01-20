
# Enhanced FaceNet for Racial Feature Classification and Deep White-Balance Editing RandomForest for Skintone Classification.ipynb

This repository contains code for building a learning model that performs race classification using deep learning techniques and skintone classification using computer vision combined with machine learning. The model is trained on racial and skin tone images and utilizes transfer learning with a pre-trained FaceNet architecture.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XMssf4Md8njSdoUCs4SzKVVWqIogwYC0?usp=sharing)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lVAQy5gHGo5i98AXbjEcMlXYEq8xhwbQ?usp=sharing)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hpdigDol1s2sjuAUD8kfdYkJTh7jP8Xe?usp=sharing


## Requirements
Make sure you have the following libraries installed:
- tensorflow >= 2.10.0
Tensorflow GPU installation example for Windows 11
``` bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow==2.10"
conda install cudnn
# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

Tensorflow GPU installation example  for Ubuntu 20.04, 22.04

``` bash
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```
- pandas 
- matplotlib 
- seaborn 
- scikit-learn 
- tensorflow-addons
- onnx
- tf2onnx
- keras_cv
- deepface
- pytorch-ignite
## File Structure
`models` directory:
- `convert_onnx.py`: convert h5 file model to onnx format.
- `Oversampling_augmented_larger_dataset_Pre_trained_model_FaceNet_Race_Classification.ipynb`, `Deep_White_Balance_Editing_RandomForest_Skintone_Classification.ipynb`, `Resize_and_Segment_Skin.ipynb`, `Fusion_answer_race_skin_tone.ipynb`: local version of the colab links above. This is the Python notebook for preprocessing data, training, evaluating and testing the model.

`weights` directory: contains weights for `RACE_CLASSIFIER_MODEL_PATH` and `SKIN_TONE_CLASSIFIER_MODEL_PATH` in "face_analysis/RaceSkintoneClassifier.py"

## Weights
 - [model_race_ptFaceNet_lr0.001_bs64_cw_oversampling_augmented_largerDataset.onnx](https://drive.google.com/file/d/1azqeUbJ4IHFv7zGuZnSbODZlxpY1gCp0/view)
 - [model_race_ptFaceNet_lr0.001_bs64_cw_oversampling_augmented_largerDataset.h5](https://drive.google.com/file/d/1-DpnGYwTo3R3jw4DF9DscTgJBnsXN4gx/view)
 - [random_forest_classifier.h5](https://drive.google.com/file/d/1mIINLy9n3Q7K-i2T6OW8Twy4VV5972eh/view)
 - [random_forest_model.onnx](https://drive.google.com/file/d/1-0_Xe2eRdnavgK9SakgHQYSOJ34H8wyp/view)

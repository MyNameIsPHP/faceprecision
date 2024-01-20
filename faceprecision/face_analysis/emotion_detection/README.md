
# Attention Mechanism & Spatial Transformer for Emotion Detection

This repository contains code for building a task learning model that performs emotion classification apply attention mechanism and spatial attention techniques. The model is trained on facial images and utilizes using CNN architecture with attention mechanism and spatial attention.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L7Mwfz3I3hxQulJqTVF1UP7K3JSn4Gds?usp=sharing)


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

## File Structure
`models` directory:
- `convert_onnx.py`: convert h5 file model to onnx format.
- `EmotionClassification.ipynb`: local version of the colab links above. This is the Python notebook for training, evaluating and testing the model.

`weights` directory: contains weights, pretrained models for `EMOTION_CLASSIFIER_MODEL_PATH` in "face_analysis/FaceAnalyzer.py"

## Weights
 - [emotion_64.onnx](https://drive.google.com/file/d/1kWFfS6H26nkYWSwt958ANC1wcUO2DYCy/view?usp=sharing)
 - [emotion_64.h5](https://drive.google.com/file/d/1iosFHFtpQIoEN_b0B7zCZ9enFnx99GDu/view?usp=sharing)

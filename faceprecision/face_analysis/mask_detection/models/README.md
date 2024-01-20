
# Attention Mechanism & Spatial Transformer for Mask Detection

This repository contains code for building a task learning model that performs mask classification using CNN with attention mechanism and spatial attention techniques. The model is trained on facial images and utilizes CNN with attention mechanism and spatial attention architecture.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P36CQtWgpDYprzvIdHTw1UWtOoQwYBkE?usp=sharing)


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
- `MaskedClassification.ipynb`: local version of the colab links above. This is the Python notebook for training, evaluating and testing the model.

`weights` directory: contains weights, pretrained models for `MASK_CLASSIFIER_MODEL_PATH` in "face_analysis/FaceAnalyzer.py"

## Weights
 - [mask_64.onnx](https://drive.google.com/file/u/4/d/1WD74hmMY8-NWR0Lw7uqk0w7iKwK8yhj_/view?usp=sharing)
 - [mask_64.h5](https://drive.google.com/file/d/1E-TU3mv1HKZtxShHD9VBCAz0Ge2HL0jP/view)

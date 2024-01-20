
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
- `multitask_learning_resnet50.ipynb`: local version of the colab links above. This is the Python notebook for training, evaluating and testing the model.

`weights` directory: contains weights, pretrained models for `AGE_GENDER_CLASSIFIER_MODEL_PATH` in "face_analysis/FaceAnalyzer.py"

## Weights
 - [age_gender_224.onnx](https://drive.google.com/file/d/1Dvmu0DS91fvRKA0LmMgOH171IpzNyhX6/view?usp=sharing)
 - [age_gender_224.h5](https://drive.google.com/file/d/1hSCh85RTw1_j6b43Z4TzL5W86pYI2kIG/view?usp=sharing)
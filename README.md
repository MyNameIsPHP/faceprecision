# faceprecision 
Faceprecision is a comprehensive face analysis project leveraging advanced deep learning and computer vision techniques. This project includes a modules and source codes for detecting and analyzing various facial attributes such as age, emotion, gender, the presence of a mask, race and skin tone.

## Features
- Face Detection: Utilize state-of-the-art YOLOv8 and non-face classification models for accurate face detection.
- Face Analysis: Leverage Multi-task Learning with Transfer Learning model for detecting age, emotion, gender, mask usage, and race/skintone.


## Installation & Usage
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
- Install Tensorflow GPU (you can follow the instruction on [tensorflow.org](https://www.tensorflow.org/install/pip)) or example scripts below:
```shell
### For Windows 10/11
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow==2.10"
conda install cudnn
# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### For Ubuntu 20.04, 22.04
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

- Install other requirements:
    ```shell
    pip install -r requirements.txt
    ```
- Then you can import the package and use its functionalities like `demo.py`:
    ```shell
    from faceprecision.FacePrecision import FacePrecision

    faceprecision = FacePrecision()
    ```
### Predict output from image
```shell
result = faceprecision.predict("test.jpg", save_path="result.jpg")
```

<p align="center"><img src="https://github.com/MyNameIsPHP/faceprecision/blob/main/result.jpg?raw=true" width="95%" height="95%"></p>


### Predict output from video
```shell
faceprecision.predict("test_video.mp4", save_path="processed_video.mp4")
```


### Real-Time Analysis: Perform face analysis in real-time using webcam
```
faceprecision.start_webcam()
```

## How to train the Multitask Attention face analysis model

You can follow the steps in `notebook.py` or this Google Colab link below
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MMTe7XPvvVuweAxBcc_erHNV1haWoH4S?usp=sharing)

Before training, download the pretrained DeepFace model by execute the command:
```shell
python download_stuff.py https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 facenet512_weights.h5
```

After training, you can convert the `.h5` model to `.onnx` and place the weights file in `weights` folder. 

```shell
python convert_to_onnx.py multitask_attention_akatsuki.h5

```

If you want to choose which model to run, you can configure the FacePrecision instance:


```shell
from faceprecision.FacePrecision import FacePrecision

faceprecision = FacePrecision(
    detector_model='your_face_detection_model.pt',
    analyzer_model='your_face_analysis_model.onnx'
)
```
By default, the `analyzer_model` will be `multitask_attention_akatsuki.onnx`, you can download the weight in the section below

## Weights
Multitask Attention Network face analysis weights
 - [multitask_attention_akatsuki.onnx](https://drive.google.com/file/d/1Jo90XwO_2iUUOJc2Q9kc6qi88EWXwLDH/view?usp=drive_link)
 - [multitask_attention_akatsuki.h5](https://drive.google.com/file/d/1-Zz8cFJJ0WTHWE124zqmjs-bNZcZnbAx/view?usp=drive_link)

YOLOv8 Face detection weights:
 - [yolov8_face_detection](https://drive.google.com/file/d/1ZwBlKsjtHAsJrxnX9obIFgTQ61noCzeM/view?usp=drive_link)

## Contributing
We welcome contributions! Please submit pull requests for improvements, bug fixes, or enhancements.



## References
 - [ultralytics](https://github.com/ultralytics/)
 - [OpenCV](https://opencv.org/)
 

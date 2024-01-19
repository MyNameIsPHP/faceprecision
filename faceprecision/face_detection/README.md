
# Face Detection with YOLOv8 with Blurriness Filter


This Python code is a script for real-time face detection using the YOLO (You Only Look Once) object detection model, filters out non-faces by checking the bluriness of detected faces, implemented using the Ultralytics library and OpenCV. 

## Features
- Real-time face detection using the YOLOv8 model.
- Filtering out non-faces based on aspect ratio.
- Bluriness check for detected faces.


## Installation

- Place weight files (ex: `yolov8_face_detection`) to `weights` folder (the download links are provided in the below section).
- Install `ultralytics` on your environment

```bash
  pip install ultralytics
```

    

## Demo

- Change current directory to "faceprecision" (`cd faceprecision`) and run `FaceDetector.py` to test the inference of YOLOv8 face detection on webcam.

```bash
  python FaceDetector.py
```



## Weights
 - [yolov8_face_detection](https://drive.google.com/file/d/1ZwBlKsjtHAsJrxnX9obIFgTQ61noCzeM/view?usp=drive_link)



## References
 - [ultralytics](https://github.com/ultralytics/)
 - [OpenCV](https://opencv.org/)
 
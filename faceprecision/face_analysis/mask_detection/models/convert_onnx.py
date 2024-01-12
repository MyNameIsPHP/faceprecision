#!pip install -U tf2onnx

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import onnx
import tf2onnx

input_model = 'mask_detector.hdf5'


# Load the trained model
model = tf.keras.models.load_model(input_model)


onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11, output_path=input_model.split(".")[0]+".onnx")
output_names = [n.name for n in onnx_model.graph.output]
print("Output names: ", output_names)


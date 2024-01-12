#!pip install -U tf2onnx

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import onnx
import tf2onnx

# Define hyperparameters
batch_size = 32
input_shape = (32, 32, 3)  # Adjust the input shape as needed

# No data augmentation for test set
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
data_folder = 'non_face_detector_samp_1/'
input_model = 'face_classification_model_32_97.h5'

# Create a test data generator
test_generator = test_datagen.flow_from_directory(
    data_folder + 'Test',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary'
)

# Load the trained model
model = tf.keras.models.load_model(input_model)


onnx_model, _ = tf2onnx.convert.from_keras(model, opset=11, output_path=input_model.replace(".h5", ".onnx"))
output_names = [n.name for n in onnx_model.graph.output]
print("Output names: ", output_names)


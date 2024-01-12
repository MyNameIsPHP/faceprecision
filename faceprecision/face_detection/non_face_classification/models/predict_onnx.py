# pip install onnxruntime


import onnxruntime
import cv2

# Load the ONNX model
onnx_model_path = 'face_classification_model_32_97.onnx'
session = onnxruntime.InferenceSession(onnx_model_path)


import numpy as np

image_path = 'test3.jpg'  # Replace with the path to your image


img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(img.shape)
# print(img)
# cv2.imshow("hi", img)
# cv2.waitKey(0)

img = cv2.resize(img, (32, 32))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)


# Get the input name from the model
input_name = session.get_inputs()[0].name

# Create an input dictionary
input_dict = {input_name: img_array}

# Perform inference
output = session.run(None, input_dict)
print(output)


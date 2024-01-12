import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# test -> [[0.00026797]]
# test2 -> 9.354949e-05
# test3 -> [[0.7477716]]


# Load the saved best model
model = tf.keras.models.load_model('face_classification_model_32_97.h5')

# Load and preprocess the single image
image_path = 'test3.jpg'  # Replace with the path to your image

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (32, 32))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make predictions on the image
predictions = model.predict(img_array)
print(predictions)
# Interpret the prediction
if predictions[0][0] > 0.5:
    print("It's a face!")
else:
    print("It's not a face.")


import cv2
import numpy as np
import onnxruntime


class NonFaceClassifier:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess_image(self, img):
        """Converts the image to the required format for the model."""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (32, 32))
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, img):
        """Makes a prediction if the image is a non-face."""
        preprocessed_img = self.preprocess_image(img)
        input_dict = {self.input_name: preprocessed_img}
        output = self.session.run(None, input_dict)
        return output[0][0] > 0.5




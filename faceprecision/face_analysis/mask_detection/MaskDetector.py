import cv2
import numpy as np
import onnxruntime

class MaskDetector:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Mapping of predicted class indices to mask status
        self.mask_mapping = {
            0: 'Masked',
            1: 'Unmasked'
        }

    def preprocess_image(self, img):
        """Converts the image to the required format for the model."""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (64, 64))
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, img):
        """Makes a prediction for the mask status in the image."""
        preprocessed_img = self.preprocess_image(img)
        input_dict = {self.input_name: preprocessed_img}
        output = self.session.run(None, input_dict)
        predicted_class = np.argmax(output)

        # Map the predicted class index to mask status
        predicted_mask_status = self.mask_mapping[predicted_class]

        return predicted_mask_status




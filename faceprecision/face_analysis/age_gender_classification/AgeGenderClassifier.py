import cv2
import numpy as np
import onnxruntime

class AgeGenderClassifier:
    def __init__(self, model_path, input_size=(224, 224)):
        self.input_size = input_size
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.age_labels = ["20-30s", "40-50s", "Baby", "Kid", "Senior", "Teenager"]


    def preprocess_image(self, img):
        """Converts the image to the required format for the model."""
        img = cv2.resize(img, self.input_size)
        img = img.astype('float32') / 255
        # Expand dimensions to add the batch size
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img):
        """Makes a prediction if the image is a non-face."""
        preprocessed_img = self.preprocess_image(img)
        input_dict = {self.input_name: preprocessed_img}
        age_pred, gender_pred = self.session.run(None, input_dict)
        predicted_age = self.age_labels[np.argmax(age_pred[0])]
        predicted_gender = ("Female" if int(gender_pred[0][0] > 0.5) == 1 else "Male")

        return [predicted_age, predicted_gender]




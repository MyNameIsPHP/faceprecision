import cv2
import onnxruntime
import numpy as np

class RaceSkintoneClassifier:
    def __init__(self, race_classifier_model_path, skintone_classifier_model_path):
        self.race_session = onnxruntime.InferenceSession(race_classifier_model_path)
        self.skintone_session = onnxruntime.InferenceSession(skintone_classifier_model_path)
        self.race_input_name = self.race_session.get_inputs()[0].name
        self.skintone_input_name = self.skintone_session.get_inputs()[0].name

        self.race_mapping = {0: 'Caucasian', 1: 'Mongoloid', 2: 'Negroid'}
        self.skintone_mapping = {0: 'dark', 1: 'light', 2: 'mid-dark', 3: 'mid-light'}

    def preprocess_image_facenet(self, image_path):
        image_size = (160, 160)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)

    def predict_skintone(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None, "Unknown"

        flattened_image = image.flatten()
        hist_r, _ = np.histogram(image[:, :, 0], bins=256, range=(0, 255))
        hist_g, _ = np.histogram(image[:, :, 1], bins=256, range=(0, 255))
        hist_b, _ = np.histogram(image[:, :, 2], bins=256, range=(0, 255))
        feature_vector = np.concatenate((hist_r, hist_g, hist_b))
        feature_vector = feature_vector.astype(np.float32).reshape(1, -1)

        skintone_output = self.skintone_session.run(None, {'float_input': feature_vector})
        return skintone_output[0], self.skintone_mapping.get(np.argmax(skintone_output[0]), "Unknown")

    def predict(self, image_path):
        preprocessed_img_race = self.preprocess_image_facenet(image_path)
        output_race = self.race_session.run(None, {self.race_input_name: preprocessed_img_race})
        predicted_race_idx = np.argmax(output_race[0])
        predicted_race = self.race_mapping.get(predicted_race_idx, "Unknown")

        skintone_probabilities, predicted_skintone = self.predict_skintone(image_path)

        # Adjust skintone probabilities if race is Negroid
        if predicted_race == 'Negroid':
            adjustments = [0.4739, 0.0162, 0.5785, 0.1693]
            skintone_probabilities = np.multiply(skintone_probabilities, adjustments)
            predicted_skintone = self.skintone_mapping.get(np.argmax(skintone_probabilities), "Unknown")

        return predicted_race, predicted_skintone

# Example of how the class would be used
# classifier = RaceSkintoneClassifier('path_to_race_model.onnx', 'path_to_skintone_model.onnx')
# race, skintone = classifier.predict('path_to_image.jpg')

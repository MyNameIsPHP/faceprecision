import cv2
import numpy as np
import onnxruntime

class EmotionDetector:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # Mapping of predicted class indices to emotions
        self.emotion_mapping = {
            0: 'Anger',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happiness',
            4: 'Neutral',
            5: 'Sadness',
            6: 'Surprise'
        }

    def preprocess_image(self, img):
        """Converts the image to the required format for the model."""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(rgb_img, (64, 64))  # Adjusted to match the model's input size
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, img):
        """Makes a prediction for the emotion in the image."""
        preprocessed_img = self.preprocess_image(img)
        input_dict = {self.input_name: preprocessed_img}
        output = self.session.run(None, input_dict)
        predicted_class = np.argmax(output)

        # Map the predicted class index to emotion
        predicted_emotion = self.emotion_mapping[predicted_class]

        return predicted_emotion





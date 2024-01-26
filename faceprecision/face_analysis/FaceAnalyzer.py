import cv2
import numpy as np
import onnxruntime
import os 


class FaceAnalyzer:
    def __init__(self, method="multitask_attention_network", model_name='multitask_attention_akatsuki.onnx', input_size=(224, 224),
                 analyze_masked=True,
                 analyze_emotion=True,
                 analyze_age=True,
                 analyze_gender=True,
                 analyze_race=True,
                 analyze_skintone=True
                 ):

        self.analyze_masked = analyze_masked
        self.analyze_emotion = analyze_emotion
        self.analyze_age = analyze_age
        self.analyze_gender = analyze_gender
        self.analyze_race = analyze_race
        self.analyze_skintone = analyze_skintone

        self.input_size = input_size
        self.session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), method, 'weights', model_name))
        self.input_name = self.session.get_inputs()[0].name
        self.age_labels = ["20-30s", "40-50s", "Baby", "Kid", "Senior", "Teenager"]
        self.gender_labels = ['Male', 'Female']
        self.emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        self.masked_labels = ['Masked', 'Unmasked']
        self.race_labels = ['Caucasian', 'Mongoloid', 'Negroid']
        self.skintone_labels =  ['dark', 'light', 'mid-dark', 'mid-light']


    def preprocess_image(self, img):
        """Converts the image to the required format for the model."""
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype('float32') / 255
        # Expand dimensions to add the batch size
        img = np.expand_dims(img, axis=0)
        return img
    

    def predict(self, face_data):
        """Makes a prediction if the image is a non-face."""
        preprocessed_img = self.preprocess_image(face_data)
        input_dict = {self.input_name: preprocessed_img}
        predictions = self.session.run(None, input_dict)

        # Create a dictionary with attribute predictions
        predictions_dict = {}

        if self.analyze_age:
            predicted_age_labels = self.age_labels[np.argmax(predictions[0])]
            predictions_dict["age"] = predicted_age_labels
       
        if self.analyze_gender:
            predicted_gender_labels = self.gender_labels[int(np.round(predictions[1]))]
            predictions_dict["gender"] = predicted_gender_labels

        if self.analyze_emotion:
            predicted_emotion_labels = self.emotion_labels[np.argmax(predictions[2])]
            predictions_dict["emotion"] = predicted_emotion_labels

        if self.analyze_masked:
            predicted_masked_labels = self.masked_labels[int(np.round(predictions[3]))]
            predictions_dict["masked"] = predicted_masked_labels

        if self.analyze_race:
            predicted_race_labels = self.race_labels[np.argmax(predictions[4])]            
            predictions_dict["race"] = predicted_race_labels

        if self.analyze_skintone:
            predicted_skintone_labels = self.skintone_labels[np.argmax(predictions[5])]
            predictions_dict["skintone"] = predicted_skintone_labels

        return predictions_dict
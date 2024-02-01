import cv2
import numpy as np
import onnxruntime
import os 


class FaceRecognizer:
    RECOGNITION_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, method="pretrained_facenet", model_name='face_recognition.onnx', input_size=(224, 224)):
        self.input_size = input_size
        self.session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__), method, 'weights', model_name))
        self.input_name = self.session.get_inputs()[0].name
        self.names_map = {0: 'Akshay Kumar',
            1: 'Alexandra Daddario',
            2: 'Alia Bhatt',
            3: 'Amitabh Bachchan',
            4: 'Andy Samberg',
            5: 'Anushka Sharma',
            6: 'Billie Eilish',
            7: 'Brad Pitt',
            8: 'Camila Cabello',
            9: 'Charlize Theron',
            10: 'Claire Holt',
            11: 'Courtney Cox',
            12: 'Dwayne Johnson',
            13: 'Elizabeth Olsen',
            14: 'Ellen Degeneres',
            15: 'Henry Cavill',
            16: 'Hieu Map',
            17: 'Hrithik Roshan',
            18: 'Hugh Jackman',
            19: 'Jessica Alba',
            20: 'Kashyap',
            21: 'Lisa Kudrow',
            22: 'Margot Robbie',
            23: 'Marmik',
            24: 'Minh Dat',
            25: 'N Lord Q Ting',
            26: 'Natalie Portman',
            27: 'Phuc Phan',
            28: 'Priyanka Chopra',
            29: 'Robert Downey Jr',
            30: 'Roger Federer',
            31: 'The Anh',
            32: 'Tom Cruise',
            33: 'Vijay Deverakonda',
            34: 'Virat Kohli',
            35: 'Zac Efron'}


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

        result_name = self.names_map[int(np.argmax(predictions))]
        confidence_score = np.max(predictions)
        result_confidence = "{:.2f}%".format(np.max(predictions) * 100)
        if confidence_score < self.RECOGNITION_CONFIDENCE_THRESHOLD:
            result_name = 'Unknown'
        return f'{result_name}: {result_confidence}'
import cv2
from concurrent.futures import ThreadPoolExecutor

class FaceAnalyzer:
    def __init__(self, analyze_mask=False, analyze_emotion=False, analyze_age_gender=False, analyze_race_skintone = False):
        # Initialize all detectors
        self.attributes = {}
        if analyze_mask:
            self.attributes['masked'] = self._init_mask_detector()
        if analyze_emotion:
            self.attributes['emotion'] = self._init_emotion_detector()
        if analyze_age_gender:
            self.attributes['age_gender'] = self._init_age_gender_classifier()
        if analyze_race_skintone:
            self.attributes['race_skintone'] = self._init_race_skintone_classifier()

    def _init_mask_detector(self):
        from .mask_detection.MaskDetector import MaskDetector
        MASK_DETECTOR_MODEL_PATH = 'face_analysis/mask_detection/weights/mask_detector.onnx'
        return MaskDetector(MASK_DETECTOR_MODEL_PATH)

    def _init_emotion_detector(self):
        from .emotion_detection.EmotionDetector import EmotionDetector
        EMOTION_DETECTOR_MODEL_PATH = 'face_analysis/emotion_detection/weights/emotion_detector.onnx'
        return EmotionDetector(EMOTION_DETECTOR_MODEL_PATH)
    
    def _init_age_gender_classifier(self):
        from .age_gender_classification.AgeGenderClassifier import AgeGenderClassifier
        AGE_GENDER_CLASSIFIER_MODEL_PATH = 'face_analysis/age_gender_classification/weights/age_gender_224.onnx'
        return AgeGenderClassifier(AGE_GENDER_CLASSIFIER_MODEL_PATH, input_size=(224, 224))
    
    def _init_race_skintone_classifier(self):
        from .race_skintone_classification.RaceSkintoneClassifier import RaceSkintoneClassifier
        RACE_CLASSIFCATION_MODEL_PATH = 'face_analysis/race_skintone_classification/weights/race_classification.onnx'
        SKINTONE_CLASSIFCATION_MODEL_PATH = 'face_analysis/race_skintone_classification/weights/skintone_classification.onnx'
        return RaceSkintoneClassifier(RACE_CLASSIFCATION_MODEL_PATH, SKINTONE_CLASSIFCATION_MODEL_PATH)
    
    def analyze_face(self, face_data):
        face = cv2.cvtColor(face_data, cv2.COLOR_BGR2RGB)
        results = {}

        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor for each detector
            futures = {detector: executor.submit(self.attributes[detector].predict, face) for detector in self.attributes}

            # Retrieve results
            for detector, future in futures.items():
                results[detector] = future.result()

        return results


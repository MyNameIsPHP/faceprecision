import cv2
from ultralytics import YOLO
from .non_face_classification.NonFaceDetector import NonFaceClassifier

class FaceDetector:
    CONFIDENCE_THRESHOLD = 0.47
    NMS_THRESHOLD = 0.4
    MODEL_PATH = "face_detection/yolov8/weights/yolov8m-face.pt"
    NON_FACE_CLASSIFIER_MODEL_PATH = 'face_detection/non_face_classification/weights/face_classification_model_32_97.onnx'
    
    def __init__(self):
        self.model = YOLO(self.MODEL_PATH)
        self.non_face_classifier = NonFaceClassifier(self.NON_FACE_CLASSIFIER_MODEL_PATH)

    @staticmethod
    def is_ratio_acceptable(height, width):
        """
        Calculate the ratio between height and width. 
        If height/width is less than 1/5 or if width/height is less than 1/5, return False.
        Otherwise, return True.
        """
        return not (height / width < 1/5 or width / height < 1/5)

    def process_image(self, img):
        """
        Process an image to detect faces.
        """
        if img is None:
            return None

        results = self.model(img)
        boxes = results[0].boxes
        confidences = results[0].boxes.conf

        rects, confidence_scores = self._prepare_rectangles(boxes, confidences)
        indices = cv2.dnn.NMSBoxes(rects, confidence_scores, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

        return self._filter_faces(img, rects, indices)

    def _prepare_rectangles(self, boxes, confidences):
        """
        Prepare rectangle coordinates and confidence scores from detection results.
        """
        rects = []
        confidence_scores = []
        for box, confidence in zip(boxes, confidences):
            x, y, x_max, y_max = box.xyxy.tolist()[0]
            rects.append([x, y, x_max - x, y_max - y])
            confidence_scores.append(float(confidence))
        return rects, confidence_scores

    def _filter_faces(self, img, rects, indices):
        """
        Filter out non-faces and apply ratio check.
        """
        result = []
        for i in indices:
            x, y, w, h = rects[i]
            if self.is_ratio_acceptable(h, w):
                face = img[int(y):int(y + h), int(x):int(x + w)]
                if not self.non_face_classifier.predict(face):
                    result.append(rects[i])
        return result


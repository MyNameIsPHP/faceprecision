import cv2
from ultralytics import YOLO
import time

class FaceDetector:
    CONFIDENCE_THRESHOLD = 0.47
    NMS_THRESHOLD = 0.4
    MODEL_PATH = "face_detection/yolov8/weights/example_yolov8_weights.pt"
    
    def __init__(self):
        self.model = YOLO(self.MODEL_PATH)

    @staticmethod
    def is_ratio_acceptable(height, width):
        """
        Calculate the ratio between height and width. 
        If height/width is less than 1/5 or if width/height is less than 1/5, return False.
        Otherwise, return True.
        """
        return not (height / width < 1/5 or width / height < 1/5)
    
    def _is_face_blurry(self, face):
        """
        Check if a detected face is blurry.
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance_of_laplacian < 5  # Threshold value, can be adjusted based on requirements
    
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
                if not self._is_face_blurry(face):
                    result.append(rects[i])
        return result


if __name__ == "__main__":
    # Initialize the FaceDetector
    face_detector = FaceDetector()

    # Open a connection to the webcam (you may need to change the device index)
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    frame_count = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        frame_count += 1

        # Process the frame with the face detector
        faces = face_detector.process_image(frame)

        # Draw rectangles around detected faces
        for face_rect in faces:
            x, y, w, h = face_rect
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # Calculate and print FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with detected faces
        cv2.imshow("Face Detection", frame)

        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

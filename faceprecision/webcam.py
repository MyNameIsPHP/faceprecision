import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from face_analysis.FaceAnalyzer import FaceAnalyzer
from face_detection.FaceDetector import FaceDetector

class WebcamFacePrecision:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_analyzer = FaceAnalyzer(analyze_mask=True, analyze_emotion=True, analyze_gender=True, analyze_age=True, analyze_race_skintone = True)
        self.fps = 0
        self.prev_frame_time = 0

    @staticmethod
    def _display_text(frame, bbox, analysis_results):
        x, y, w, h = bbox
        text_color = (255, 255, 255)  # White color for text
        text_background_color = (0, 0, 0)  # Black background for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        padding = 3
        spacing = 7
        display_list = []

        for i, result in enumerate(analysis_results):
            if (isinstance(result, str)):
                display_list.append(result)

            elif (isinstance(result, list)):
                display_list.extend(result)

        for i, text in enumerate(display_list):            
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = int(x + w + 7)
            text_y = int(y + padding + (text_size[1] + padding) * i + spacing*i)
            # Draw background for text
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - padding), (text_x + text_size[0] + padding, text_y + padding), text_background_color, -1)
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_rectangle(frame, bbox):
        x, y, w, h = bbox
        box_color = (0, 255, 0)  # Green color for boxes
        box_thickness = 2
        # Draw the bounding box
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), box_color, box_thickness)

    
    @staticmethod
    def _draw_corners(frame, bbox, analysis_results):
        x, y, w, h = bbox
        corner_color = (0, 255, 0)  # Green color for corners
        corner_length = 20  # Length of each corner line
        corner_thickness = 3  # Thickness of corner lines
        # Other variables (text_color, text_background_color, etc.) remain the same

        # Function to draw a single corner
        def draw_corner(ix, iy, is_horizontal):
            ix, iy = int(ix), int(iy)  # Ensure the coordinates are integers
            if is_horizontal:
                cv2.line(frame, (ix, iy), (ix + corner_length, iy), corner_color, corner_thickness)
            else:
                cv2.line(frame, (ix, iy), (ix, iy + corner_length), corner_color, corner_thickness)

        # Draw the four corners
        # Top-left corner
        draw_corner(x, y, True)
        draw_corner(x, y, False)

        # Top-right corner
        draw_corner(x + w - corner_length, y, True)
        draw_corner(x + w , y, False)

        # Bottom-left corner
        draw_corner(x, y + h, True)
        draw_corner(x, y + h - corner_length, False)

        # Bottom-right corner
        draw_corner(x + w - corner_length, y + h, True)
        draw_corner(x + w, y + h - corner_length, False)
        
    def _update_fps(self, frame):
        new_frame_time = time.time()
        self.fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def _process_frame(self, frame, executor):
        detected_bboxes = self.face_detector.process_image(frame)
        futures = [(executor.submit(self.face_analyzer.analyze_face, frame[int(y):int(y + h), int(x):int(x + w)]), (x, y, w, h)) for (x, y, w, h) in detected_bboxes]

        for future, bbox in futures:
            result_list = future.result().values()
            self._draw_corners(frame, bbox, result_list)
            # self._draw_rectangle(frame, bbox, result_list)
            self._display_text(frame, bbox, result_list)

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        with ThreadPoolExecutor(max_workers=8) as executor:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self._process_frame(frame, executor)
                self._update_fps(frame)

                cv2.imshow('Face Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_detector = WebcamFacePrecision()
    webcam_detector.start_webcam()

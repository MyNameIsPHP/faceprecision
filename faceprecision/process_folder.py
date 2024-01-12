import os
import time
import cv2
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from face_analysis.FaceAnalyzer import FaceAnalyzer
from face_detection.FaceDetector import FaceDetector

class FolderProcessor:
    def __init__(self, data_folder_path, output_folder_name = "output", csv_file_path = "answer.csv", plot_result = False, save_result_images = False, save_by_num_of_faces = False, display_text = False, save_cropped_faces = False):
        
        self.folder_path = data_folder_path
        self.output_folder_name = output_folder_name
        self.csv_file_path = csv_file_path
        self.save_result_images = save_result_images
        self.save_by_num_of_faces = save_by_num_of_faces
        self.display_text = display_text
        self.save_cropped_faces = save_cropped_faces
        self.plot_result = plot_result
        self.face_detector = FaceDetector()
        self.face_analyzer = FaceAnalyzer(analyze_mask=True, analyze_emotion=True, analyze_gender=True, analyze_age=True, analyze_race_skintone = True)
        
        os.makedirs(self.output_folder_name, exist_ok=True)

        # Load file_name_to_image_id.json
        with open('file_name_to_image_id.json', 'r') as file:
            self.filename_to_id = json.load(file)

        # Initialize CSV file with headers
        with open(os.path.join(output_folder_name, self.csv_file_path), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Define your CSV headers here based on the analysis results
            writer.writerow(["file_name", "bbox", "image_id", "race", "age", "emotion", "gender", "skintone", "masked"])

        if self.save_cropped_faces:
            self.cropped_faces_folder = os.path.join(self.output_folder_name, "cropped_faces")
            os.makedirs(self.cropped_faces_folder, exist_ok=True)

    def _plot_result(self, frame, bbox, analysis_results):
        x, y, w, h = bbox
        box_color = (0, 255, 0)  # Green color for boxes
        box_thickness = 2
        # Draw the bounding box
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), box_color, box_thickness)
    
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
            if (self.display_text):
                # Draw background for text
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - padding), (text_x + text_size[0] + padding, text_y + padding), text_background_color, -1)
                # Draw text
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def is_image_file(filename):
        ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
        return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

    def _process_frame(self, frame, executor, original_filename):
        detected_bboxes = self.face_detector.process_image(frame)
        futures = [(executor.submit(self.face_analyzer.analyze_face, frame[int(y):int(y + h), int(x):int(x + w)]), (x, y, w, h)) for (x, y, w, h) in detected_bboxes]
        original_frame = frame.copy()
        
        for future, bbox in futures:
            analysis_results = future.result()
            result_list = analysis_results.values()

            x, y, w, h = bbox
            if self.save_cropped_faces:
                cropped_face = original_frame[int(y):int(y + h), int(x):int(x + w)]
                cropped_face_filename = f"{original_filename}_face_{int(x)}_{int(y)}_{result_list}.jpg"
                cv2.imwrite(os.path.join(self.cropped_faces_folder, cropped_face_filename), cropped_face)

            image_id = self.filename_to_id.get(original_filename, "Unknown")
            with open(os.path.join(self.output_folder_name, self.csv_file_path), mode='a', newline='') as file:
                writer = csv.writer(file)

                # Assuming `analysis_results` is a dictionary, you can format it like this:
                writer.writerow([original_filename, str(list(bbox)), image_id, analysis_results.get("race_skintone")[0], analysis_results.get("age"), analysis_results.get("emotion"), analysis_results.get("gender"), analysis_results.get("race_skintone")[1], analysis_results.get("masked")])

            frame = self._plot_result(frame, bbox, result_list)

        if self.save_result_images:
            if (self.save_by_num_of_faces):
                save_path = os.path.join(self.output_folder_name, f"images/{len(detected_bboxes)}/")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path + original_filename, frame)
            else:
                save_path = os.path.join(self.output_folder_name, "images/")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path + original_filename, frame)
                
            
        if (self.plot_result):
            cv2.imshow("Image", frame)
            cv2.waitKey(0)

    def process(self):
        if not os.path.exists(self.folder_path):
            print(f"The folder '{self.folder_path}' does not exist.")
            return
        
        start_time = time.time()

        image_paths = [os.path.join(self.folder_path, file) for file in os.listdir(self.folder_path) if self.is_image_file(file)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            for path in image_paths:
                img = cv2.imread(path)
                self._process_frame(img, executor, os.path.basename(path))
        
        end_time = time.time()
        # Calculate the total inference time
        total_inference_time = end_time - start_time
        print("Total processing time:", total_inference_time)

if __name__ == "__main__":
    webcam_detector = FolderProcessor("more_than_1", plot_result = False, save_result_images = True, save_by_num_of_faces = False, display_text = True, save_cropped_faces = True)
    webcam_detector.process()

import os
import time
import cv2
import csv
import pandas as pd
import json
import ast  # For converting string representation of list to actual list
from concurrent.futures import ThreadPoolExecutor
from face_analysis.FaceAnalyzer import FaceAnalyzer
from face_detection.FaceDetector import FaceDetector

class Evaluator:
    def __init__(self, csv_file_path = "answer.csv", data_folder_path = "data", output_folder_name = "evaluation_output",  plot_result = False, save_wrong_cases = False):
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.data_folder_path = data_folder_path        
        self.output_folder_name = output_folder_name
        self.plot_result = plot_result
        self.face_detector = FaceDetector()
        self.face_analyzer = FaceAnalyzer(analyze_mask=True, analyze_emotion=True, analyze_gender=True, analyze_age=True, analyze_race_skintone = True)
        self.save_wrong_cases = save_wrong_cases
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total_iou = 0
        self.total_faces = 0
        self.correct_masked_predictions = 0
        self.total_masked_predictions = 0
        self.correct_emotion_predictions = 0
        self.total_emotion_predictions = 0
        self.correct_gender_predictions = 0
        self.total_gender_predictions = 0
        self.correct_age_predictions = 0
        self.total_age_predictions = 0
        self.correct_race_predictions = 0
        self.total_race_predictions = 0
        self.correct_skintone_predictions = 0
        self.total_skintone_predictions = 0
    

        os.makedirs(self.output_folder_name, exist_ok=True)
        if (save_wrong_cases):
            os.makedirs(os.path.join(self.output_folder_name, "wrong_face_detection", "original"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_face_detection", "result"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_non_face", "not_face"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_non_face", "face"), exist_ok=True)

            os.makedirs(os.path.join(self.output_folder_name, "wrong_masked", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_masked", "faces"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_emotion", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_emotion", "faces"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_gender", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_gender", "faces"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_age", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_age", "faces"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_race", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_race", "faces"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_skintone", "images"), exist_ok=True)
            os.makedirs(os.path.join(self.output_folder_name, "wrong_skintone", "faces"), exist_ok=True)

        # Load file_name_to_image_id.json
        with open('file_name_to_image_id.json', 'r') as file:
            self.filename_to_id = json.load(file)

    @staticmethod
    def _calculate_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[0] + box1[2], box2[0] + box2[2])
        y2_inter = min(box1[1] + box1[3], box2[1] + box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area

        return iou
    
    def _update_classification_counts(self, detected_bboxes, gt_bboxes):
            """
            Update counts of TP, FP, and FN based on IoU threshold.
            """
            matched = set()
            for detected_box in detected_bboxes:
                is_match = False
                for idx, gt_box in enumerate(gt_bboxes):
                    if idx in matched:
                        continue
                    iou = self._calculate_iou(detected_box, gt_box)
                    if iou > 0.5:  # IoU threshold
                        matched.add(idx)
                        is_match = True
                        break
                if is_match:
                    self.tp += 1
                else:
                    self.fp += 1

            self.fn += len(gt_bboxes) - len(matched)
    
    def _calculate_precision_recall_ap(self):
        """
        Calculate Precision, Recall, and Average Precision.
        """
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        # Note: AP calculation would typically involve integrating over a curve of precision and recall values at different thresholds.
        # For simplicity, this example uses precision at a single IoU threshold. Consider using a more robust method for real applications.
        ap = precision  # Simplified AP calculation
        return precision, recall, ap

    def _process_frame(self, frame, executor, original_filename, gt_bboxes, gt_races, gt_ages, gt_emotions, gt_genders, gt_skintones, gt_maskeds):
        detected_bboxes = self.face_detector.process_image(frame)
        self._update_classification_counts(detected_bboxes, gt_bboxes)
        original_frame = frame.copy()

        for (x, y, w, h) in detected_bboxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        for (x, y, w, h) in gt_bboxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        if (self.save_wrong_cases and len(detected_bboxes) != len(gt_bboxes)):
            cv2.imwrite(os.path.join(self.output_folder_name, "wrong_face_detection", "original", original_filename), original_frame)
            cv2.imwrite(os.path.join(self.output_folder_name, "wrong_face_detection", "result", original_filename), frame)
        
        matched = []
        ordered_choosen_detected_bboxes = []
        wrong_detection = []

        index_map = {}
        for i, detected_box in enumerate(detected_bboxes):
            best_iou = 0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_bboxes):
                iou = self._calculate_iou(detected_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > 0.5:
                matched.append((detected_box, gt_bboxes[best_gt_idx]))
                ordered_choosen_detected_bboxes.append(detected_box)
                index_map[i] = best_gt_idx
            else:
                wrong_detection.append(detected_box)
        
        # Print the results
        print(f"File: {original_filename}, Matched Boxes: {matched}, Wrong Detections: {wrong_detection}")
        for i, box in enumerate(wrong_detection):
            x, y, w, h = box
            face = original_frame[int(y):int(y + h), int(x):int(x + w)]

            # Save the cropped face
            face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{i}.jpg'
            if (self.save_wrong_cases):
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_non_face", "not_face", face_filename), face)

        futures = [(executor.submit(self.face_analyzer.analyze_face, frame[int(y):int(y + h), int(x):int(x + w)]), (x, y, w, h)) for (x, y, w, h) in ordered_choosen_detected_bboxes]
        
        for i, match in enumerate(matched):
            x, y, w, h = match[0]
            face = original_frame[int(y):int(y + h), int(x):int(x + w)]

            # Save the cropped face
            face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{i}.jpg'
            if (self.save_wrong_cases):
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_non_face", "face", face_filename), face)

            self.total_iou += self._calculate_iou(match[0], match[1])
            self.total_faces += 1

        idx = 0
        for future, bbox in futures:
            analysis_results = future.result()
            result_list = analysis_results.values()
            predicted_race = analysis_results.get("race_skintone")[0]
            predicted_age = analysis_results.get("age")
            predicted_emotion = analysis_results.get("emotion")
            predicted_gender =  analysis_results.get("gender")
            predicted_skintone = analysis_results.get("race_skintone")[1]
            predicted_masked = analysis_results.get("masked")

            gt_race = gt_races[index_map[idx]]
            if predicted_race == gt_race:
                self.correct_race_predictions += 1
            elif (self.save_wrong_cases):
                face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{idx}_{gt_race}_{predicted_race}.jpg'
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_race", "faces", face_filename), face)
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_race", "images", original_filename), original_frame)
            self.total_race_predictions += 1

            gt_age = gt_ages[index_map[idx]]
            if predicted_age == gt_age:
                self.correct_age_predictions += 1
            elif (self.save_wrong_cases):
                face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{idx}_{gt_age}_{predicted_age}.jpg'
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_age", "faces", face_filename), face)
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_age", "images", original_filename), original_frame)
            self.total_age_predictions += 1

            gt_emotion = gt_emotions[index_map[idx]]
            if predicted_emotion == gt_emotion:
                self.correct_emotion_predictions += 1
            elif (self.save_wrong_cases):
                face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{idx}_{gt_emotion}_{predicted_emotion}.jpg'
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_emotion", "faces", face_filename), face)
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_emotion", "images", original_filename), original_frame)
            self.total_emotion_predictions += 1

            gt_gender = gt_genders[index_map[idx]]
            if predicted_gender == gt_gender:
                self.correct_gender_predictions += 1
            elif (self.save_wrong_cases):
                face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{idx}_{gt_gender}_{predicted_gender}.jpg'
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_gender", "faces", face_filename), face)
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_gender", "images", original_filename), original_frame)
            self.total_gender_predictions += 1

            gt_skintone = gt_skintones[index_map[idx]]
            if predicted_skintone == gt_skintone:
                self.correct_skintone_predictions += 1
            elif (self.save_wrong_cases):
                face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{idx}_{gt_skintone}_{predicted_skintone}.jpg'
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_skintone", "faces", face_filename), face)
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_skintone", "images", original_filename), original_frame)
            self.total_skintone_predictions += 1
            
            gt_masked = gt_maskeds[index_map[idx]]
            if predicted_masked == gt_masked:
                self.correct_masked_predictions += 1
            elif (self.save_wrong_cases):
                face_filename = os.path.splitext(os.path.basename(original_filename))[0] + f'_{idx}_{gt_masked}_{predicted_masked}.jpg'
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_masked", "faces", face_filename), face)
                cv2.imwrite(os.path.join(self.output_folder_name, "wrong_masked", "images", original_filename), original_frame)
            self.total_masked_predictions += 1

            idx += 1

        if (self.plot_result):
            cv2.imshow("Image", frame)
            cv2.waitKey(0)


    def process(self):
        if not os.path.exists(self.data_folder_path):
            print(f"The folder '{self.data_folder_path}' does not exist.")
            return
        
        if not os.path.exists(self.csv_file_path):
            print(f"The file '{self.csv_file_path}' does not exist.")
            return


        with ThreadPoolExecutor(max_workers=8) as executor:
         
            # Group the data by 'file_name' and iterate through each group
            for file_name, group in self.df.groupby('file_name'):
                
                # Count the number of faces in the current file
                num_faces = len(group)

                # Get the bounding boxes for the current file
                bboxes = group['bbox'].tolist()
                gt_races = group['race'].tolist()  # Assuming 'emotion' is the column name
                gt_ages = group['age'].tolist()  # Assuming 'emotion' is the column name
                gt_emotions = group['emotion'].tolist()  # Assuming 'emotion' is the column name
                gt_genders = group['gender'].tolist()  # Assuming 'emotion' is the column name
                gt_skintones = group['skintone'].tolist()  # Assuming 'emotion' is the column name
                gt_maskeds = group['masked'].tolist()  # Assuming 'emotion' is the column name

                
                # Print the file name, number of faces, and bounding boxes
                print(f"File: {file_name}, Number of Faces: {num_faces}, Bounding Boxes: {bboxes}")
                img = cv2.imread(os.path.join(self.data_folder_path, file_name))

                # Convert ground truth boxes from string to list
                gt_bboxes = [ast.literal_eval(bbox) for bbox in bboxes]

                self._process_frame(img, executor, file_name, 
                                    gt_bboxes = gt_bboxes, 
                                    gt_races = gt_races,
                                    gt_ages = gt_ages,
                                    gt_emotions = gt_emotions,
                                    gt_genders = gt_genders, 
                                    gt_skintones = gt_skintones, 
                                    gt_maskeds = gt_maskeds)
                
        precision, recall, ap = self._calculate_precision_recall_ap()
        print(f"Precision: {precision}, Recall: {recall}, AP: {ap}")
        # Since only one class (face), mAP is the same as AP
        print(f"mAP: {ap}")
        print(f"Average IoU: {0 if self.total_iou == 0 else self.total_faces / self.total_iou}")
        print(f"Race Classification Accuracy: {0 if self.total_race_predictions == 0 else self.correct_race_predictions / self.total_race_predictions}")
        print(f"Age Classification Accuracy: {0 if self.total_age_predictions == 0 else self.correct_age_predictions / self.total_age_predictions}")
        print(f"Emotion Classification Accuracy: {0 if self.total_emotion_predictions == 0 else self.correct_emotion_predictions / self.total_emotion_predictions}")
        print(f"Gender Classification Accuracy: {0 if self.total_gender_predictions == 0 else self.correct_gender_predictions / self.total_gender_predictions}")
        print(f"Skintone Classification Accuracy: {0 if self.total_skintone_predictions == 0 else self.correct_skintone_predictions / self.total_skintone_predictions}")
        print(f"Masked Classification Accuracy: {0 if self.total_masked_predictions == 0 else self.correct_masked_predictions / self.total_masked_predictions}")

if __name__ == "__main__":
    webcam_detector = Evaluator("labels.csv", "data", "evaluation_output", plot_result = False, save_wrong_cases = False)
    webcam_detector.process()

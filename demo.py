from faceprecision.FacePrecision import FacePrecision

if __name__ == "__main__":
    faceprecision = FacePrecision(
        detector_method='yolov8',
        detector_model='example_yolov8_weights.pt',
        analyzer_method='multitask_attention_network',
        analyzer_model='multitask_attention_akatsuki.onnx',
        recognizer_method='pretrained_facenet',
        recognizer_model='face_recognition.onnx'
    )

    faceprecision.start_webcam()
    # result = faceprecision.predict("example_images/test.jpg", save_path="example_images/result_test.jpg")
    # faceprecision.predict("example_videos/test_video.mp4", save_path="example_videos/processed_video.mp4")

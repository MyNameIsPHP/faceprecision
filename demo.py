from faceprecision.FacePrecision import FacePrecision

if __name__ == "__main__":
    faceprecision = FacePrecision(
        detector_method='yolov8',
        detector_model='example_yolov8_weights.pt',
        analyzer_method='multitask_attention_network',
        analyzer_model='multitask_attention_akatsuki.onnx'
    )

    faceprecision.start_webcam()
    # result = faceprecision.predict("test.jpg", save_path="result.jpg")
    # faceprecision.predict("test_video.mp4", save_path="processed_video.mp4")

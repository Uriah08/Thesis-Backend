# import os
# from ultralytics import YOLO

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO_DETECTION_PATH = os.path.join(BASE_DIR, 'model', 'yolo', 'yolov8-2.pt')
# YOLO_CLASSIFICATION_PATH = os.path.join(BASE_DIR, 'model', 'yolocls', 'yolov8-cls-1.pt')  # YOLOv8 classification model

# print("="*50)
# print("LOADING AI MODELS...")
# print("="*50)

# # Load YOLO detection model
# detection_model = YOLO(YOLO_DETECTION_PATH)
# print("✓ YOLO detection model loaded")

# # Load YOLO classification model
# classification_model = YOLO(YOLO_CLASSIFICATION_PATH)
# print("✓ YOLO classification model loaded")

# print("="*50)
# print("✓ ALL MODELS READY!")
# print("="*50)

import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_DETECTION_PATH = os.path.join(BASE_DIR, 'model', 'yolo', 'yolov8-2.pt')
YOLO_CLASSIFICATION_PATH = os.path.join(BASE_DIR, 'model', 'yolocls', 'yolov8-cls-1.pt')

# Don't load models at import time!
_detection_model = None
_classification_model = None

def get_detection_model():
    """Lazy load detection model"""
    global _detection_model
    if _detection_model is None:
        print("Loading YOLO detection model...")
        _detection_model = YOLO(YOLO_DETECTION_PATH)
        print("✓ YOLO detection model loaded")
    return _detection_model

def get_classification_model():
    """Lazy load classification model"""
    global _classification_model
    if _classification_model is None:
        print("Loading YOLO classification model...")
        _classification_model = YOLO(YOLO_CLASSIFICATION_PATH)
        print("✓ YOLO classification model loaded")
    return _classification_model

# Export functions instead of models
detection_model = get_detection_model
classification_model = get_classification_model
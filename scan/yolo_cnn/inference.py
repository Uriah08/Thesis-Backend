import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_DETECTION_PATH = os.path.join(BASE_DIR, 'model', 'yolo', 'yolov8-2.pt')
YOLO_CLASSIFICATION_PATH = os.path.join(BASE_DIR, 'model', 'yolocls', 'yolov8-cls-1.pt')  # YOLOv8 classification model

print("="*50)
print("LOADING AI MODELS...")
print("="*50)

# Load YOLO detection model
detection_model = YOLO(YOLO_DETECTION_PATH)
print("✓ YOLO detection model loaded")

# Load YOLO classification model
classification_model = YOLO(YOLO_CLASSIFICATION_PATH)
print("✓ YOLO classification model loaded")

print("="*50)
print("✓ ALL MODELS READY!")
print("="*50)
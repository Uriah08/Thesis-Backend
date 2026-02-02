import os
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'yolo', 'yolov8-2.pt')
CNN_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cnn', 'best_mobilenetv2_compatible.h5')

print("="*50)
print("LOADING AI MODELS...")
print("="*50)

# Load models when module is imported
detection_model = YOLO(YOLO_MODEL_PATH)
print("✓ YOLO model loaded")

classification_model = keras.models.load_model(CNN_MODEL_PATH, compile=False)
print("✓ MobileNet model loaded")

print("="*50)
print("✓ ALL MODELS READY!")
print("="*50)
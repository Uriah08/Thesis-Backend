import os
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo', 'yolov8-1.pt')
CNN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cnn', 'best_mobilenetv2_model-2.keras')

detection_model = YOLO(YOLO_MODEL_PATH)
classification_model = keras.models.load_model(CNN_MODEL_PATH)
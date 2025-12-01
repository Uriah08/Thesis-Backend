# myapp/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication

import cv2
import os
import shutil
import numpy as np
from django.conf import settings

from .yolo_cnn.inference import detection_model, classification_model

from datetime import datetime
import uuid

class ScanView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def detect_and_classify(self, image):
        original = image.copy()
        results = detection_model(image, conf=0.25, verbose=False)

        detections = []
        dry_count = 0
        undried_count = 0
        reject_count = 0

        COLORS = {
            'DRY': (0, 255, 0),
            'UNDRIED': (0, 165, 255),
            'REJECT': (0, 0, 255)
        }

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_name = detection_model.names[int(box.cls[0])]

            if class_name.lower() == 'fish':
                crop = original[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (224, 224))
                crop_norm = np.expand_dims(crop_resized / 255.0, axis=0)

                pred = classification_model.predict(crop_norm, verbose=0)[0][0]

                if pred > 0.5:
                    label = 'UNDRIED'
                    conf_pct = float(pred * 100)
                    color = COLORS['UNDRIED']
                    undried_count += 1
                else:
                    label = 'DRY'
                    conf_pct = float((1 - pred) * 100)
                    color = COLORS['DRY']
                    dry_count += 1

            elif class_name.lower() == 'reject':
                label = 'REJECT'
                conf_pct = float(confidence * 100)
                color = COLORS['REJECT']
                reject_count += 1
            else:
                continue

            cv2.rectangle(original, (x1, y1), (x2, y2), color, 3)
            label_text = f"{label} {conf_pct:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(original, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
            cv2.putText(original, label_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            detections.append({
                'class': class_name,
                'label': label,
                'confidence': conf_pct,
                'bbox': [x1, y1, x2, y2]
            })

        return original, detections, dry_count, undried_count, reject_count

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"detail": "No image uploaded"}, status=400)

        # Save uploaded image temporarily
        temp_path = "temp.jpg"
        with open(temp_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        image = cv2.imread(temp_path)

        annotated_image, detections, dry_count, undried_count, reject_count = self.detect_and_classify(image)

        # Create per-user folder
        user_id = str(request.user.id) if request.user.is_authenticated else "default"
        user_folder = os.path.join(settings.MEDIA_ROOT, "scanned", user_id)

        # Delete all previous scans for this user
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)

        os.makedirs(user_folder, exist_ok=True)

        # Save new annotated image
        filename = f"{uuid.uuid4().hex[:6]}.jpg"
        save_path = os.path.join(user_folder, filename)
        cv2.imwrite(save_path, annotated_image)

        # Build URL
        image_url = request.build_absolute_uri(
            settings.MEDIA_URL + f"scanned/{user_id}/{filename}?t={datetime.now().timestamp()}"
        )

        return Response({
            'image_url': image_url,
            'detections': detections,
            'dry_count': dry_count,
            'undried_count': undried_count,
            'reject_count': reject_count,
            'total': dry_count + undried_count + reject_count
        })
















# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.authentication import TokenAuthentication

# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from ultralytics import YOLO
# import cv2
# import os
# import numpy as np
# from tempfile import NamedTemporaryFile

# import uuid
# from datetime import datetime
# import glob

# class ScanView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]

#     # =======================
#     # Load Models Once
#     # =======================
#     yolo_model = YOLO("models/yolo/yolov8-1.pt")

#     cnn_model = models.resnet18(pretrained=False)
#     cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
#     cnn_model.load_state_dict(torch.load("models/cnn/resnet18-1.pth", map_location="cpu"))
#     cnn_model.eval()

#     dryness_labels = ['FULLY_DRY', 'ALMOST_DRY', 'PARTIALLY_DRY', 'WET']

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # =======================
#     # Detect + Classify
#     # =======================
#     def detect_and_classify(self, image_path):
#         image = cv2.imread(image_path)
#         results = self.yolo_model.predict(source=image_path, conf=0.5, verbose=False)[0]

#         class_names = results.names
#         detections = []

#         colors = {
#             "FULLY_DRY": (79, 255, 79),        # green
#             "ALMOST_DRY": (77, 246, 255),     # yellow
#             "PARTIALLY_DRY": (71, 190, 255),  # orange
#             "WET": (255, 175, 79),              # blue
#             "REJECT": (41, 41, 255)           # red
#         }

#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             cls_name = class_names[cls_id]
#             conf = float(box.conf[0])

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = image[y1:y2, x1:x2]

#             if crop.size == 0:
#                 continue

#             # 🐟 Classify fish dryness
#             if cls_name.lower() == "fish":
#                 input_tensor = self.transform(crop).unsqueeze(0)
#                 with torch.no_grad():
#                     outputs = self.cnn_model(input_tensor)
#                     _, predicted = torch.max(outputs, 1)
#                     label = self.dryness_labels[predicted.item()]
#                     color = colors[label]

#             # 🚫 Reject fish
#             elif cls_name.lower() == "reject":
#                 label = "REJECT"
#                 color = colors["REJECT"]

#             detections.append({
#                 "class": cls_name,
#                 "label": label,
#                 "confidence": round(conf, 2),
#                 "box": [x1, y1, x2, y2]
#             })

#             # Draw box
#             box_thickness = 5
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

#             # Label
#             text_scale = 1.5
#             text_thickness = 3
#             font = cv2.FONT_HERSHEY_SIMPLEX

#             (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
#             label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
#             label_bg_bottom_right = (x1 + text_width + 10, y1)

#             cv2.rectangle(image, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
#             cv2.putText(image, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

#         return image, detections

#     # =======================
#     # POST (Upload Image)
#     # =======================
#     def post(self, request):
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # ✅ Save uploaded file temporarily
#         temp_path = "temp.jpg"
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # ✅ Run detection
#         output, detections = self.detect_and_classify(temp_path)
#         print(detections)

#         # ✅ Make sure predictions directory exists
#         os.makedirs("media/predictions", exist_ok=True)

#         # ✅ Delete previous predictions (optional)
#         for old_file in glob.glob("media/predictions/*.jpg"):
#             os.remove(old_file)

#         # ✅ Create new annotated filename
#         filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
#         output_path = os.path.join("media", "predictions", filename)

#         # ✅ Save the annotated image
#         cv2.imwrite(output_path, output)

#         # ✅ Convert path to URL-friendly format (use forward slashes)
#         relative_path = output_path.replace("\\", "/")

#         # ✅ Build correct absolute URL for response
#         image_url = request.build_absolute_uri("/" + relative_path)

#         return Response({
#             "image_url": image_url,
#             "detections": detections
#         })

# import os
# import uuid
# import cv2
# import numpy as np
# from django.conf import settings
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status
# from .yolo_cnn.inference import detection_model, classification_model

# # Updated CNN classes
# CLASS_NAMES = ['DRY', 'UNDRIED']
# IMG_SIZE = 224
# OUTPUT_DIR = os.path.join(settings.MEDIA_ROOT, 'scans')
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def crop_and_preprocess(img, box):
#     x1, y1, x2, y2 = map(int, box)
#     crop_img = img[y1:y2, x1:x2]
#     crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
#     crop_img = crop_img / 255.0
#     crop_img = np.expand_dims(crop_img, axis=0)
#     return crop_img

# class ScanView(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request, *args, **kwargs):
#         image_file = request.FILES.get('image', None)
#         if not image_file:
#             return Response({"error": "No image uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         # Convert uploaded file to OpenCV image
#         img_array = np.frombuffer(image_file.read(), np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         if img is None:
#             return Response({"error": "Invalid image."}, status=status.HTTP_400_BAD_REQUEST)

#         # YOLO detection
#         results = detection_model.predict(img)
#         detections = results[0].boxes.data.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()

#         response_detections = []

#         for i, cls in enumerate(classes):
#             class_name = results[0].names[int(cls)]
#             box = detections[i][:4]
#             confidence = float(detections[i][4])

#             detection_data = {
#                 "class": class_name,
#                 "confidence": confidence,
#                 "box": box.tolist()
#             }

#             if class_name.lower() == 'fish':
#                 # CNN dryness classification
#                 crop_img = crop_and_preprocess(img, box)
#                 pred = classification_model.predict(crop_img)
#                 pred_class = CLASS_NAMES[np.argmax(pred)]
#                 pred_conf = float(np.max(pred))

#                 detection_data["dryness_class"] = pred_class
#                 detection_data["dryness_confidence"] = pred_conf

#                 # Draw CNN label
#                 cv2.putText(img, f"{pred_class} {pred_conf:.2f}", 
#                             (int(box[0]), int(box[3]+20)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

#             # Draw YOLO box
#             color = (0, 255, 0) if class_name.lower() == 'fish' else (0, 0, 255)
#             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
#             cv2.putText(img, f"{class_name} {confidence:.2f}", 
#                         (int(box[0]), int(box[1]-10)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             response_detections.append(detection_data)

#         # Save annotated image
#         filename = f"{uuid.uuid4().hex}.jpg"
#         save_path = os.path.join(OUTPUT_DIR, filename)
#         cv2.imwrite(save_path, img)
#         image_url = os.path.join(settings.MEDIA_URL, 'scans', filename)

#         return Response({
#             "image_url": image_url,
#             "detections": response_detections
#         }, status=status.HTTP_200_OK)

# import os
# import uuid
# import cv2
# import numpy as np
# from django.conf import settings
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status
# from tensorflow import keras
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from .yolo_cnn.inference import detection_model  # YOLO detection

# # Binary classification
# CLASS_NAMES = ['DRY', 'UNDRIED']
# IMG_SIZE = 224

# # CNN model path
# CNN_MODEL_PATH = os.path.join(settings.BASE_DIR, 'scan', 'yolo_cnn', 'best_mobilenetv2_binary_model.keras')
# classification_model = keras.models.load_model(CNN_MODEL_PATH)

# # Output folder for annotated images
# OUTPUT_DIR = os.path.join(settings.MEDIA_ROOT, 'scans')
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ===== Preprocessing function =====
# def crop_and_preprocess(img, box):
#     x1, y1, x2, y2 = map(int, box)
#     crop_img = img[y1:y2, x1:x2]
#     crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
#     crop_img = preprocess_input(crop_img)  # MobileNetV2 preprocessing
#     crop_img = np.expand_dims(crop_img, axis=0)
#     return crop_img

# # ===== Django API View =====
# class ScanView(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request, *args, **kwargs):
#         image_file = request.FILES.get('image', None)
#         if not image_file:
#             return Response({"error": "No image uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         # Convert uploaded file to OpenCV image
#         img_array = np.frombuffer(image_file.read(), np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         if img is None:
#             return Response({"error": "Invalid image."}, status=status.HTTP_400_BAD_REQUEST)

#         # YOLO detection
#         results = detection_model.predict(img)
#         detections = results[0].boxes.data.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()

#         response_detections = []

#         for i, cls in enumerate(classes):
#             class_name = results[0].names[int(cls)]
#             box = detections[i][:4]
#             confidence = float(detections[i][4])

#             detection_data = {
#                 "class": class_name,
#                 "confidence": confidence,
#                 "box": box.tolist()
#             }

#             if class_name.lower() == 'fish':
#                 # CNN dryness classification
#                 crop_img = crop_and_preprocess(img, box)
#                 pred = classification_model.predict(crop_img)
                
#                 # Map sigmoid output to DRY/UNDRIED
#                 pred_value = float(pred[0][0])
#                 pred_class = CLASS_NAMES[1] if pred_value >= 0.5 else CLASS_NAMES[0]
#                 pred_conf = pred_value if pred_class == 'UNDRIED' else 1 - pred_value

#                 detection_data["dryness_class"] = pred_class
#                 detection_data["dryness_confidence"] = pred_conf

#                 # Draw CNN label
#                 cv2.putText(img, f"{pred_class} {pred_conf:.2f}", 
#                             (int(box[0]), int(box[3]+20)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

#             # Draw YOLO box
#             color = (0, 255, 0) if class_name.lower() == 'fish' else (0, 0, 255)
#             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
#             cv2.putText(img, f"{class_name} {confidence:.2f}", 
#                         (int(box[0]), int(box[1]-10)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             response_detections.append(detection_data)

#         # Save annotated image
#         filename = f"{uuid.uuid4().hex}.jpg"
#         save_path = os.path.join(OUTPUT_DIR, filename)
#         cv2.imwrite(save_path, img)
#         image_url = os.path.join(settings.MEDIA_URL, 'scans', filename)

#         return Response({
#             "image_url": image_url,
#             "detections": response_detections
#         }, status=status.HTTP_200_OK)

# import os
# import uuid
# import cv2
# import glob
# from datetime import datetime
# from django.conf import settings
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status
# from .yolo_cnn.inference import detect_and_classify

# # Make sure the predictions folder exists
# PREDICTIONS_DIR = os.path.join(settings.MEDIA_ROOT, 'predictions')
# os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# class ScanView(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request):
#         # 1️⃣ Get uploaded image
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # 2️⃣ Save uploaded file temporarily
#         temp_path = os.path.join(settings.MEDIA_ROOT, "temp.jpg")
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # 3️⃣ Run detection + classification
#         output_img, detections = detect_and_classify(temp_path)

#         # 4️⃣ Delete previous predictions (optional)
#         for old_file in glob.glob(os.path.join(PREDICTIONS_DIR, "*.jpg")):
#             os.remove(old_file)

#         # 5️⃣ Save annotated image
#         filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
#         output_path = os.path.join(PREDICTIONS_DIR, filename)
#         cv2.imwrite(output_path, output_img)

#         # 6️⃣ Build URL for frontend
#         relative_path = os.path.join(settings.MEDIA_URL, 'predictions', filename).replace("\\", "/")
#         image_url = request.build_absolute_uri(relative_path)

#         # 7️⃣ Return JSON
#         return Response({
#             "image_url": image_url,
#             "detections": detections
#         }, status=status.HTTP_200_OK)

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.parsers import MultiPartParser, FormParser
# import cv2
# import numpy as np
# import base64
# from .yolo_cnn.inference import detection_model, classification_model

# class CheckModelsView(APIView):
#     """Check if models are loaded"""
    
#     def get(self, request):
#         try:
#             return Response({
#                 'success': True,
#                 'message': 'Models are loaded and ready',
#                 'models': {
#                     'yolo': 'loaded',
#                     'mobilenet': 'loaded'
#                 }
#             })
#         except Exception as e:
#             return Response({
#                 'success': False,
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class ScanView(APIView):
#     """Process fish image detection and classification"""
    
#     parser_classes = (MultiPartParser, FormParser)
    
#     def post(self, request):
#         try:
#             # Get image from request
#             image_file = request.FILES.get('image')
            
#             if not image_file:
#                 return Response({
#                     'success': False,
#                     'error': 'No image provided'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # Read and decode image
#             image_bytes = image_file.read()
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if image is None:
#                 return Response({
#                     'success': False,
#                     'error': 'Invalid image format'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # Process image
#             result = self.process_image(image)
            
#             return Response(result)
            
#         except Exception as e:
#             return Response({
#                 'success': False,
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def process_image(self, image):
#         """Process image with YOLO and MobileNet"""
        
#         original = image.copy()
        
#         # YOLO Detection
#         results = detection_model(image, conf=0.25, verbose=False)
        
#         # Counters
#         dry_count = 0
#         undried_count = 0
#         reject_count = 0
#         detections = []
        
#         # Colors (BGR)
#         COLORS = {
#             'DRY': (0, 255, 0),
#             'UNDRIED': (0, 165, 255),
#             'REJECT': (0, 0, 255)
#         }
        
#         # Process each detection
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             class_name = detection_model.names[int(box.cls[0])]
            
#             if class_name.lower() == 'fish':
#                 # Crop and classify
#                 crop = original[y1:y2, x1:x2]
#                 crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#                 crop_resized = cv2.resize(crop_rgb, (224, 224))
#                 crop_norm = np.expand_dims(crop_resized / 255.0, axis=0)
                
#                 # Predict dryness
#                 pred = classification_model.predict(crop_norm, verbose=0)[0][0]
                
#                 if pred > 0.5:
#                     label = 'UNDRIED'
#                     conf_pct = float(pred * 100)
#                     color = COLORS['UNDRIED']
#                     undried_count += 1
#                 else:
#                     label = 'DRY'
#                     conf_pct = float((1 - pred) * 100)
#                     color = COLORS['DRY']
#                     dry_count += 1
                
#                 # Draw bounding box
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
#                 # Draw label
#                 label_text = f"{label} {conf_pct:.1f}%"
#                 (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#                 cv2.rectangle(image, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
#                 cv2.putText(image, label_text, (x1+5, y1-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
#                 detections.append({
#                     'type': label,
#                     'confidence': conf_pct,
#                     'bbox': [x1, y1, x2, y2]
#                 })
                
#             elif class_name.lower() == 'reject':
#                 reject_count += 1
#                 color = COLORS['REJECT']
                
#                 # Draw bounding box
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
#                 # Draw label
#                 label_text = f"REJECT {confidence*100:.1f}%"
#                 (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#                 cv2.rectangle(image, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
#                 cv2.putText(image, label_text, (x1+5, y1-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
#                 detections.append({
#                     'type': 'REJECT',
#                     'confidence': float(confidence * 100),
#                     'bbox': [x1, y1, x2, y2]
#                 })
        
#         # Add summary overlay
#         summary = [
#             f"DRY: {dry_count}",
#             f"UNDRIED: {undried_count}",
#             f"REJECT: {reject_count}"
#         ]
        
#         y_offset = 30
#         for i, text in enumerate(summary):
#             (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#             cv2.rectangle(image, (10, y_offset+i*40-th-5), (tw+20, y_offset+i*40+5), (0,0,0), -1)
#             cv2.putText(image, text, (15, y_offset+i*40),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
#         # Convert annotated image to base64
#         _, buffer = cv2.imencode('.jpg', image)
#         img_base64 = base64.b64encode(buffer).decode('utf-8')
        
#         return {
#             'success': True,
#             'dry_count': dry_count,
#             'undried_count': undried_count,
#             'reject_count': reject_count,
#             'total': dry_count + undried_count + reject_count,
#             'detections': detections,
#             'annotated_image': f'data:image/jpeg;base64,{img_base64}'
#         }

# myapp/views.py
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.parsers import MultiPartParser, FormParser
# import os
# import cv2
# import numpy as np
# import uuid
# from django.conf import settings
# from .yolo_cnn.inference import detection_model, classification_model


# class CheckModelsView(APIView):
#     """Check if models are loaded"""
    
#     def get(self, request):
#         try:
#             return Response({
#                 'success': True,
#                 'message': 'Models are loaded and ready',
#                 'models': {
#                     'yolo': 'loaded',
#                     'mobilenet': 'loaded'
#                 }
#             })
#         except Exception as e:
#             return Response({
#                 'success': False,
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class ScanView(APIView):
#     """Process fish image detection and classification"""
    
#     parser_classes = (MultiPartParser, FormParser)
    
#     def post(self, request):
#         try:
#             image_file = request.FILES.get('image')
            
#             if not image_file:
#                 return Response({
#                     'success': False,
#                     'error': 'No image provided'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # Read image bytes
#             image_bytes = image_file.read()
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if image is None:
#                 return Response({
#                     'success': False,
#                     'error': 'Invalid image format'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # Process the image
#             result, saved_relative_path = self.process_image(image)
            
#             # Construct full URL for frontend
#             saved_url = request.build_absolute_uri(settings.MEDIA_URL + saved_relative_path)
#             result['saved_url'] = saved_url
            
#             return Response(result)
            
#         except Exception as e:
#             return Response({
#                 'success': False,
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def process_image(self, image):
#         """Run YOLO detection, classify, annotate, and save image"""
        
#         original = image.copy()
#         results = detection_model(image, conf=0.25, verbose=False)
        
#         dry_count = 0
#         undried_count = 0
#         reject_count = 0
#         detections = []
        
#         COLORS = {
#             'DRY': (0, 255, 0),
#             'UNDRIED': (0, 165, 255),
#             'REJECT': (0, 0, 255)
#         }
        
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             class_name = detection_model.names[int(box.cls[0])]
            
#             if class_name.lower() == 'fish':
#                 # Crop and classify
#                 crop = original[y1:y2, x1:x2]
#                 crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#                 crop_resized = cv2.resize(crop_rgb, (224, 224))
#                 crop_norm = np.expand_dims(crop_resized / 255.0, axis=0)
                
#                 pred = classification_model.predict(crop_norm, verbose=0)[0][0]
                
#                 if pred > 0.5:
#                     label = 'UNDRIED'
#                     conf_pct = float(pred * 100)
#                     color = COLORS['UNDRIED']
#                     undried_count += 1
#                 else:
#                     label = 'DRY'
#                     conf_pct = float((1 - pred) * 100)
#                     color = COLORS['DRY']
#                     dry_count += 1
                
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
#                 label_text = f"{label} {conf_pct:.1f}%"
#                 (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#                 cv2.rectangle(image, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
#                 cv2.putText(image, label_text, (x1+5, y1-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
#                 detections.append({
#                     'type': label,
#                     'confidence': conf_pct,
#                     'bbox': [x1, y1, x2, y2]
#                 })
                
#             elif class_name.lower() == 'reject':
#                 reject_count += 1
#                 color = COLORS['REJECT']
#                 cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
#                 label_text = f"REJECT {confidence*100:.1f}%"
#                 (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#                 cv2.rectangle(image, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
#                 cv2.putText(image, label_text, (x1+5, y1-5),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
#                 detections.append({
#                     'type': 'REJECT',
#                     'confidence': float(confidence * 100),
#                     'bbox': [x1, y1, x2, y2]
#                 })
        
#         # Summary overlay
#         summary = [
#             f"DRY: {dry_count}",
#             f"UNDRIED: {undried_count}",
#             f"REJECT: {reject_count}"
#         ]
        
#         y_offset = 30
#         for i, text in enumerate(summary):
#             (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#             cv2.rectangle(image, (10, y_offset+i*40-th-5), (tw+20, y_offset+i*40+5), (0,0,0), -1)
#             cv2.putText(image, text, (15, y_offset+i*40),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
#         # Save annotated image
#         scanned_dir = os.path.join(settings.MEDIA_ROOT, 'scanned')
#         os.makedirs(scanned_dir, exist_ok=True)
        
#         filename = f"{uuid.uuid4().hex}.jpg"
#         save_path = os.path.join(scanned_dir, filename)
#         cv2.imwrite(save_path, image)
        
#         result_dict = {
#             'success': True,
#             'dry_count': dry_count,
#             'undried_count': undried_count,
#             'reject_count': reject_count,
#             'total': dry_count + undried_count + reject_count,
#             'detections': detections
#         }
        
#         return result_dict, os.path.join('scanned', filename)  # relative path for MEDIA_URL







































# class ScanView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]

#     # =======================
#     # Load Models Once
#     # =======================
#     yolo_model = YOLO("models/yolo/yolov8-1.pt")

#     # Load CNN model with checkpoint format
#     checkpoint = torch.load("models/cnn/resnet18-2.pth", map_location="cpu")
#     cnn_model = models.resnet18(pretrained=False)
#     cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
    
#     # Handle both old and new checkpoint formats
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         # New format (from your training script)
#         cnn_model.load_state_dict(checkpoint['model_state_dict'])
#         dryness_labels = checkpoint.get('classes', ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET'])
#     else:
#         # Old format (direct state dict)
#         cnn_model.load_state_dict(checkpoint)
#         dryness_labels = ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET']
    
#     cnn_model.eval()

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # =======================
#     # Detect + Classify
#     # =======================
#     def detect_and_classify(self, image_path):
#         image = cv2.imread(image_path)
#         results = self.yolo_model.predict(source=image_path, conf=0.5, verbose=False)[0]

#         class_names = results.names
#         detections = []

#         colors = {
#             "DRY": (79, 255, 79),              # green
#             "ALMOST_DRY": (77, 246, 255),      # yellow
#             "PARTIALLY_DRY": (71, 190, 255),   # orange
#             "WET": (255, 175, 79),             # blue
#             "REJECT": (41, 41, 255)            # red
#         }

#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             cls_name = class_names[cls_id]
#             conf = float(box.conf[0])

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = image[y1:y2, x1:x2]

#             if crop.size == 0:
#                 continue

#             # 🐟 Classify fish dryness
#             if cls_name.lower() == "fish":
#                 input_tensor = self.transform(crop).unsqueeze(0)
#                 with torch.no_grad():
#                     outputs = self.cnn_model(input_tensor)
#                     _, predicted = torch.max(outputs, 1)
#                     label = self.dryness_labels[predicted.item()]
#                     color = colors.get(label, (255, 255, 255))  # Default to white if not found

#             # 🚫 Reject fish
#             elif cls_name.lower() == "reject":
#                 label = "REJECT"
#                 color = colors["REJECT"]

#             detections.append({
#                 "class": cls_name,
#                 "label": label,
#                 "confidence": round(conf, 2),
#                 "box": [x1, y1, x2, y2]
#             })

#             # Draw box
#             box_thickness = 5
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

#             # Label
#             text_scale = 1.5
#             text_thickness = 3
#             font = cv2.FONT_HERSHEY_SIMPLEX

#             (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
#             label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
#             label_bg_bottom_right = (x1 + text_width + 10, y1)

#             cv2.rectangle(image, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
#             cv2.putText(image, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

#         return image, detections

#     # =======================
#     # POST (Upload Image)
#     # =======================
#     def post(self, request):
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # ✅ Save uploaded file temporarily
#         temp_path = "temp.jpg"
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # ✅ Run detection
#         output, detections = self.detect_and_classify(temp_path)

#         # ✅ Make sure predictions directory exists
#         os.makedirs("media/predictions", exist_ok=True)

#         # ✅ Delete previous predictions (optional)
#         for old_file in glob.glob("media/predictions/*.jpg"):
#             os.remove(old_file)

#         # ✅ Create new annotated filename
#         filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
#         output_path = os.path.join("media", "predictions", filename)

#         # ✅ Save the annotated image
#         cv2.imwrite(output_path, output)

#         # ✅ Convert path to URL-friendly format (use forward slashes)
#         relative_path = output_path.replace("\\", "/")

#         # ✅ Build correct absolute URL for response
#         image_url = request.build_absolute_uri("/" + relative_path)

#         return Response({
#             "image_url": image_url,
#             "detections": detections
#         })



#     # =======================
#     # POST (Upload Image)
#     # =======================
#     def post(self, request):
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # ✅ Create a unique temporary file for each upload
#         with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
#             for chunk in image.chunks():
#                 temp_file.write(chunk)
#             temp_path = temp_file.name

#         try:
#             # ✅ Run detection safely
#             output, detections = self.detect_and_classify(temp_path)

#             # ✅ Make sure predictions directory exists
#             os.makedirs("media/predictions", exist_ok=True)

#             # ✅ Optionally delete old predictions
#             for old_file in glob.glob("media/predictions/*.jpg"):
#                 os.remove(old_file)

#             # ✅ Create new annotated filename
#             filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
#             output_path = os.path.join("media", "predictions", filename)

#             # ✅ Save the annotated image
#             cv2.imwrite(output_path, output)

#             # ✅ Convert path to URL-friendly format (use forward slashes)
#             relative_path = output_path.replace("\\", "/")

#             # ✅ Build correct absolute URL for response
#             image_url = request.build_absolute_uri("/" + relative_path)

#             return Response({
#                 "image_url": image_url,
#                 "detections": detections
#             })

#         finally:
#             # ✅ Always clean up temp file
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)

# import os
# import uuid
# import cv2
# import numpy as np
# from django.conf import settings
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response
# from rest_framework import status
# from .yolo_cnn.inference import detection_model, classification_model

# CLASS_NAMES = ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET']
# IMG_SIZE = 224
# OUTPUT_DIR = os.path.join(settings.MEDIA_ROOT, 'scans')
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def crop_and_preprocess(img, box):
#     x1, y1, x2, y2 = map(int, box)
#     crop_img = img[y1:y2, x1:x2]
#     crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
#     crop_img = crop_img / 255.0
#     crop_img = np.expand_dims(crop_img, axis=0)
#     return crop_img

# class ScanView(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request, *args, **kwargs):
#         image_file = request.FILES.get('image', None)
#         if not image_file:
#             return Response({"error": "No image uploaded."}, status=status.HTTP_400_BAD_REQUEST)

#         # Convert uploaded file to OpenCV image
#         img_array = np.frombuffer(image_file.read(), np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         if img is None:
#             return Response({"error": "Invalid image."}, status=status.HTTP_400_BAD_REQUEST)

#         # YOLO detection
#         results = detection_model.predict(img)
#         detections = results[0].boxes.data.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()

#         response_detections = []

#         for i, cls in enumerate(classes):
#             class_name = results[0].names[int(cls)]
#             box = detections[i][:4]
#             confidence = float(detections[i][4])

#             detection_data = {
#                 "class": class_name,
#                 "confidence": confidence,
#                 "box": box.tolist()
#             }

#             if class_name.lower() == 'fish':
#                 # CNN dryness classification
#                 crop_img = crop_and_preprocess(img, box)
#                 pred = classification_model.predict(crop_img)
#                 pred_class = CLASS_NAMES[np.argmax(pred)]
#                 pred_conf = float(np.max(pred))

#                 detection_data["dryness_class"] = pred_class
#                 detection_data["dryness_confidence"] = pred_conf

#                 # Draw CNN label
#                 cv2.putText(img, f"{pred_class} {pred_conf:.2f}", 
#                             (int(box[0]), int(box[3]+20)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

#             # Draw YOLO box
#             color = (0, 255, 0) if class_name.lower() == 'fish' else (0, 0, 255)
#             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
#             cv2.putText(img, f"{class_name} {confidence:.2f}", 
#                         (int(box[0]), int(box[1]-10)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             response_detections.append(detection_data)

#         # Save annotated image
#         filename = f"{uuid.uuid4().hex}.jpg"
#         save_path = os.path.join(OUTPUT_DIR, filename)
#         cv2.imwrite(save_path, img)
#         image_url = os.path.join(settings.MEDIA_URL, 'scans', filename)

#         return Response({
#             "image_url": image_url,
#             "detections": response_detections
#         }, status=status.HTTP_200_OK)

# import os
# import glob
# import uuid
# from datetime import datetime

# import cv2
# import numpy as np
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.authentication import TokenAuthentication
# from rest_framework.response import Response

# from .yolo_cnn.inference import detection_model, classification_model

# CLASS_NAMES = ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET']
# IMG_SIZE = 224

# COLORS = {
#     "ALMOST_DRY": (77, 246, 255),     # yellow
#     "DRY": (79, 255, 79),             # green
#     "PARTIALLY_DRY": (71, 190, 255),  # orange
#     "WET": (255, 175, 79),            # blue
#     "REJECT": (41, 41, 255)           # red
# }


# class ScanView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]

#     def crop_and_preprocess(self, img, box):
#         """Crop YOLO box and prepare for CNN"""
#         x1, y1, x2, y2 = map(int, box)
#         crop_img = img[y1:y2, x1:x2]
#         crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
#         crop_img = crop_img / 255.0
#         crop_img = np.expand_dims(crop_img, axis=0)
#         return crop_img

#     def detect_and_classify(self, image_path):
#         img = cv2.imread(image_path)
#         results = detection_model.predict(img)
#         detections = results[0].boxes.data.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()
#         class_names_yolo = results[0].names

#         output_detections = []

#         for i, cls in enumerate(classes):
#             class_name = class_names_yolo[int(cls)]
#             box = detections[i][:4]
#             confidence = float(detections[i][4])

#             label = class_name.upper()
#             color = COLORS.get(label, (0, 255, 255))

#             # If fish, run CNN dryness classification
#             if class_name.lower() == "fish":
#                 crop_img = self.crop_and_preprocess(img, box)
#                 pred = classification_model.predict(crop_img)
#                 cnn_class = CLASS_NAMES[np.argmax(pred)]
#                 cnn_conf = float(np.max(pred))
#                 label = cnn_class
#                 color = COLORS[cnn_class]

#             elif class_name.lower() == "reject":
#                 label = "REJECT"
#                 color = COLORS["REJECT"]

#             output_detections.append({
#                 "class": class_name,
#                 "label": label,
#                 "confidence": round(confidence, 2),
#                 "box": box.tolist()
#             })

#             # ===== Draw bounding box like your design =====
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

#             # Draw label background
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             text_scale = 1.5
#             text_thickness = 3
#             (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
#             label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
#             label_bg_bottom_right = (x1 + text_width + 10, y1)
#             cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
#             cv2.putText(img, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

#         return img, output_detections

#     def post(self, request):
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # Save temporarily
#         temp_path = "temp.jpg"
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # Run YOLO + CNN
#         output_img, detections = self.detect_and_classify(temp_path)

#         # Save annotated image
#         os.makedirs("media/predictions", exist_ok=True)
#         for old_file in glob.glob("media/predictions/*.jpg"):
#             os.remove(old_file)

#         filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
#         output_path = os.path.join("media/predictions", filename)
#         cv2.imwrite(output_path, output_img)

#         image_url = request.build_absolute_uri("/media/predictions/" + filename)

#         return Response({
#             "image_url": image_url,
#             "detections": detections
#         })


# from django.shortcuts import render
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.authentication import TokenAuthentication

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torchvision import models
# from ultralytics import YOLO
# import cv2
# import os
# import numpy as np
# from tempfile import NamedTemporaryFile

# import uuid
# from datetime import datetime
# import glob

# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         self.activations = output.detach()
    
#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
    
#     def generate_cam(self, input_tensor, target_class):
#         # Forward pass
#         self.model.eval()
#         output = self.model(input_tensor)
        
#         # Backward pass
#         self.model.zero_grad()
#         output[0, target_class].backward()
        
#         # Get gradients and activations
#         gradients = self.gradients[0]  # [C, H, W]
#         activations = self.activations[0]  # [C, H, W]
        
#         # Calculate weights (global average pooling of gradients)
#         weights = gradients.mean(dim=(1, 2))  # [C]
        
#         # Weighted combination of activation maps
#         cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
#         for i, w in enumerate(weights):
#             cam += w * activations[i]
        
#         # Apply ReLU (only positive influence)
#         cam = F.relu(cam)
        
#         # Normalize to 0-1
#         if cam.max() > 0:
#             cam = cam - cam.min()
#             cam = cam / cam.max()
        
#         return cam.cpu().numpy()

# class ScanView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]

#     # =======================
#     # Load Models Once
#     # =======================
#     yolo_model = YOLO("models/yolo/yolov8-1.pt")

#     cnn_model = models.resnet18(pretrained=False)
#     cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
#     cnn_model.load_state_dict(torch.load("models/cnn/resnet18-1.pth", map_location="cpu"))
#     cnn_model.eval()

#     # Initialize GradCAM with ResNet18's last convolutional layer
#     gradcam = GradCAM(cnn_model, cnn_model.layer4[-1])

#     dryness_labels = ['FULLY_DRY', 'ALMOST_DRY', 'PARTIALLY_DRY', 'WET']

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # =======================
#     # Apply Grad-CAM Heatmap
#     # =======================
#     def apply_gradcam_heatmap(self, crop_image, input_tensor, predicted_class):
#         """
#         Apply Grad-CAM heatmap on the cropped fish image
#         """
#         # Generate CAM
#         cam = self.gradcam.generate_cam(input_tensor, predicted_class)
        
#         # Resize CAM to match crop size
#         h, w = crop_image.shape[:2]
#         cam_resized = cv2.resize(cam, (w, h))
        
#         # Create heatmap
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
#         # Overlay heatmap on original crop (40% heatmap, 60% original)
#         overlay = cv2.addWeighted(crop_image, 0.6, heatmap, 0.4, 0)
        
#         return overlay

#     # =======================
#     # Detect + Classify
#     # =======================
#     def detect_and_classify(self, image_path):
#         image = cv2.imread(image_path)
#         results = self.yolo_model.predict(source=image_path, conf=0.5, verbose=False)[0]

#         class_names = results.names
#         detections = []

#         colors = {
#             "FULLY_DRY": (79, 255, 79),        # green
#             "ALMOST_DRY": (77, 246, 255),     # yellow
#             "PARTIALLY_DRY": (71, 190, 255),  # orange
#             "WET": (255, 175, 79),              # blue
#             "REJECT": (41, 41, 255)           # red
#         }

#         # Store Grad-CAM images
#         gradcam_crops = []

#         for idx, box in enumerate(results.boxes):
#             cls_id = int(box.cls[0])
#             cls_name = class_names[cls_id]
#             conf = float(box.conf[0])

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = image[y1:y2, x1:x2]

#             if crop.size == 0:
#                 continue

#             # 🐟 Classify fish dryness
#             if cls_name.lower() == "fish":
#                 input_tensor = self.transform(crop).unsqueeze(0)
#                 with torch.no_grad():
#                     outputs = self.cnn_model(input_tensor)
#                     _, predicted = torch.max(outputs, 1)
#                     predicted_class = predicted.item()
#                     label = self.dryness_labels[predicted_class]
#                     color = colors[label]
                
#                 # Generate Grad-CAM visualization
#                 gradcam_overlay = self.apply_gradcam_heatmap(crop.copy(), input_tensor, predicted_class)
#                 gradcam_crops.append({
#                     "index": idx,
#                     "image": gradcam_overlay,
#                     "label": label
#                 })

#             # 🚫 Reject fish
#             elif cls_name.lower() == "reject":
#                 label = "REJECT"
#                 color = colors["REJECT"]

#             detections.append({
#                 "class": cls_name,
#                 "label": label,
#                 "confidence": round(conf, 2),
#                 "box": [x1, y1, x2, y2]
#             })

#             # Draw box
#             box_thickness = 5
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

#             # Label
#             text_scale = 1.5
#             text_thickness = 3
#             font = cv2.FONT_HERSHEY_SIMPLEX

#             (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
#             label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
#             label_bg_bottom_right = (x1 + text_width + 10, y1)

#             cv2.rectangle(image, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
#             cv2.putText(image, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

#         return image, detections, gradcam_crops

#     # =======================
#     # POST (Upload Image)
#     # =======================
#     def post(self, request):
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # ✅ Save uploaded file temporarily
#         temp_path = "temp.jpg"
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # ✅ Run detection
#         output, detections, gradcam_crops = self.detect_and_classify(temp_path)
#         print(detections)

#         # ✅ Make sure directories exist
#         os.makedirs("media/predictions", exist_ok=True)
#         os.makedirs("media/gradcam", exist_ok=True)

#         # ✅ Delete previous predictions (optional)
#         for old_file in glob.glob("media/predictions/*.jpg"):
#             os.remove(old_file)
#         for old_file in glob.glob("media/gradcam/*.jpg"):
#             os.remove(old_file)

#         # ✅ Create timestamp for filenames
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         unique_id = uuid.uuid4().hex[:6]

#         # ✅ Save the annotated image
#         annotated_filename = f"annotated_{timestamp}_{unique_id}.jpg"
#         annotated_path = os.path.join("media", "predictions", annotated_filename)
#         cv2.imwrite(annotated_path, output)

#         # ✅ Save Grad-CAM images
#         gradcam_urls = []
#         for grad_item in gradcam_crops:
#             gradcam_filename = f"gradcam_{timestamp}_{unique_id}_fish{grad_item['index']}_{grad_item['label']}.jpg"
#             gradcam_path = os.path.join("media", "gradcam", gradcam_filename)
#             cv2.imwrite(gradcam_path, grad_item['image'])
            
#             # Build URL
#             relative_path = gradcam_path.replace("\\", "/")
#             gradcam_url = request.build_absolute_uri("/" + relative_path)
#             gradcam_urls.append({
#                 "fish_index": grad_item['index'],
#                 "label": grad_item['label'],
#                 "gradcam_url": gradcam_url
#             })

#         # ✅ Convert path to URL-friendly format
#         relative_path = annotated_path.replace("\\", "/")
#         image_url = request.build_absolute_uri("/" + relative_path)

#         return Response({
#             "image_url": image_url,
#             "detections": detections,
#             "gradcam_visualizations": gradcam_urls
#         })

# import os
# import glob
# import uuid
# from datetime import datetime

# import cv2
# import numpy as np
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.authentication import TokenAuthentication
# from rest_framework.response import Response

# from .yolo_cnn.inference import detection_model, classification_model
# import tensorflow as tf

# CLASS_NAMES = ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET']
# IMG_SIZE = 224

# COLORS = {
#     "ALMOST_DRY": (77, 246, 255),     # yellow
#     "DRY": (79, 255, 79),             # green
#     "PARTIALLY_DRY": (71, 190, 255),  # orange
#     "WET": (255, 175, 79),            # blue
#     "REJECT": (41, 41, 255)           # red
# }


# class GradCAMMobileNet:
#     """Grad-CAM for TensorFlow/Keras MobileNet models"""
    
#     def __init__(self, model, layer_name=None):
#         self.model = model
        
#         # Find the last convolutional layer if not specified
#         if layer_name is None:
#             print("\n=== Searching for convolutional layers ===")
#             for layer in reversed(model.layers):
#                 # Check if layer has output attribute and it's a tensor
#                 if hasattr(layer, 'output'):
#                     try:
#                         output_shape = layer.output.shape
#                         print(f"Layer: {layer.name} - Shape: {output_shape} - Type: {layer.__class__.__name__}")
                        
#                         # Conv layer has 4D output (batch, height, width, channels)
#                         if len(output_shape) == 4:
#                             layer_name = layer.name
#                             print(f"✓ Selected layer: {layer_name}")
#                             break
#                     except:
#                         pass
                        
#                 # Alternative: check by layer class name
#                 if 'conv' in layer.__class__.__name__.lower():
#                     layer_name = layer.name
#                     print(f"✓ Selected Conv layer by name: {layer_name}")
#                     break
        
#         if layer_name is None:
#             raise ValueError("Could not find a convolutional layer in the model")
        
#         self.layer_name = layer_name
        
#         try:
#             self.grad_model = tf.keras.models.Model(
#                 inputs=model.inputs,
#                 outputs=[model.get_layer(layer_name).output, model.output]
#             )
#             print(f"✓ Grad-CAM model created successfully with layer: {layer_name}\n")
#         except Exception as e:
#             raise ValueError(f"Error creating Grad-CAM model with layer '{layer_name}': {e}")
    
#     def generate_cam(self, img_array, pred_index=None):
#         """
#         Generate Grad-CAM heatmap
#         img_array: preprocessed image (1, 224, 224, 3)
#         pred_index: target class index (if None, uses predicted class)
#         """
#         # Convert to tensor if it's a numpy array
#         if isinstance(img_array, np.ndarray):
#             img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
#         with tf.GradientTape() as tape:
#             # Watch the input
#             tape.watch(img_array)
            
#             # Forward pass - returns [conv_output, prediction]
#             outputs = self.grad_model(img_array, training=False)
            
#             # Handle different output formats
#             if isinstance(outputs, list):
#                 conv_outputs = outputs[0]
#                 predictions = outputs[1]
#             else:
#                 conv_outputs, predictions = outputs
            
#             # Ensure predictions is a tensor
#             if isinstance(predictions, list):
#                 predictions = predictions[0]
            
#             # Get pred_index
#             if pred_index is None:
#                 pred_index = tf.argmax(predictions[0])
#             else:
#                 pred_index = int(pred_index)
            
#             # Get the score for the predicted class
#             class_channel = predictions[0, pred_index]
        
#         # Get gradients of the class score with respect to the conv layer output
#         grads = tape.gradient(class_channel, conv_outputs)
        
#         # This is a vector where each entry is the mean intensity of the gradient 
#         # over a specific feature map channel
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
#         # Convert to numpy for manipulation
#         conv_outputs = conv_outputs.numpy()
#         pooled_grads = pooled_grads.numpy()
        
#         # Multiply each channel in the feature map array
#         # by "how important this channel is" with regard to the predicted class
#         for i in range(pooled_grads.shape[0]):
#             conv_outputs[0, :, :, i] *= pooled_grads[i]
        
#         # The channel-wise mean of the resulting feature map
#         # is our heatmap of class activation
#         heatmap = np.mean(conv_outputs[0], axis=-1)
        
#         # Normalize the heatmap between 0 & 1 for visualization
#         heatmap = np.maximum(heatmap, 0)  # ReLU to only keep positive influences
        
#         if heatmap.max() > 0:
#             heatmap = heatmap / heatmap.max()
        
#         return heatmap
    
#     def apply_heatmap(self, original_img, heatmap):
#         """
#         Overlay heatmap on original image
#         original_img: original crop image (H, W, 3) in BGR
#         heatmap: normalized heatmap (H', W')
#         """
#         # Resize heatmap to match image size
#         h, w = original_img.shape[:2]
#         heatmap_resized = cv2.resize(heatmap, (w, h))
        
#         # Convert to RGB colormap
#         heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
#         # Overlay: 60% original + 40% heatmap
#         overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
#         return overlay


# class ScanView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]
    
#     # Initialize Grad-CAM once (class variable)
#     gradcam = None
#     gradcam_initialized = False
    
#     @classmethod
#     def initialize_gradcam(cls):
#         """Initialize Grad-CAM once for the class"""
#         if not cls.gradcam_initialized:
#             try:
#                 print("Initializing Grad-CAM...")
#                 cls.gradcam = GradCAMMobileNet(classification_model)
#                 cls.gradcam_initialized = True
#                 print("✓ Grad-CAM initialized successfully\n")
#             except Exception as e:
#                 print(f"✗ Warning: Could not initialize Grad-CAM: {e}")
#                 cls.gradcam = None
#                 cls.gradcam_initialized = True  # Mark as attempted

#     def crop_and_preprocess(self, img, box):
#         """Crop YOLO box and prepare for CNN"""
#         x1, y1, x2, y2 = map(int, box)
#         crop_img = img[y1:y2, x1:x2]
#         crop_img_resized = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
#         crop_img_normalized = crop_img_resized / 255.0
#         crop_img_batch = np.expand_dims(crop_img_normalized, axis=0).astype(np.float32)
#         return crop_img, crop_img_resized, crop_img_batch

#     def detect_and_classify(self, image_path):
#         img = cv2.imread(image_path)
#         results = detection_model.predict(img)
#         detections = results[0].boxes.data.cpu().numpy()
#         classes = results[0].boxes.cls.cpu().numpy()
#         class_names_yolo = results[0].names

#         output_detections = []
#         gradcam_crops = []

#         for i, cls in enumerate(classes):
#             class_name = class_names_yolo[int(cls)]
#             box = detections[i][:4]
#             confidence = float(detections[i][4])

#             label = class_name.upper()
#             color = COLORS.get(label, (0, 255, 255))

#             # If fish, run CNN dryness classification
#             if class_name.lower() == "fish":
#                 # Get original crop, resized crop, and preprocessed batch
#                 crop_original, crop_resized, crop_batch = self.crop_and_preprocess(img, box)
                
#                 # Predict
#                 pred = classification_model.predict(crop_batch, verbose=0)
#                 cnn_class_idx = np.argmax(pred)
#                 cnn_class = CLASS_NAMES[cnn_class_idx]
#                 cnn_conf = float(np.max(pred))
#                 label = cnn_class
#                 color = COLORS[cnn_class]
                
#                 # Generate Grad-CAM (only if initialized successfully)
#                 if self.gradcam is not None:
#                     try:
#                         heatmap = self.gradcam.generate_cam(crop_batch, pred_index=cnn_class_idx)
#                         gradcam_overlay = self.gradcam.apply_heatmap(crop_resized, heatmap)
                        
#                         gradcam_crops.append({
#                             "index": i,
#                             "image": gradcam_overlay,
#                             "label": cnn_class
#                         })
#                         print(f"✓ Grad-CAM generated for fish {i}: {cnn_class}")
#                     except Exception as e:
#                         print(f"✗ Warning: Grad-CAM generation failed for fish {i}: {e}")
#                         import traceback
#                         traceback.print_exc()

#             elif class_name.lower() == "reject":
#                 label = "REJECT"
#                 color = COLORS["REJECT"]

#             output_detections.append({
#                 "class": class_name,
#                 "label": label,
#                 "confidence": round(confidence, 2),
#                 "box": box.tolist()
#             })

#             # ===== Draw bounding box like your design =====
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

#             # Draw label background
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             text_scale = 1.5
#             text_thickness = 3
#             (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
#             label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
#             label_bg_bottom_right = (x1 + text_width + 10, y1)
#             cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
#             cv2.putText(img, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

#         return img, output_detections, gradcam_crops

#     def post(self, request):
#         # Initialize Grad-CAM on first request
#         if not ScanView.gradcam_initialized:
#             ScanView.initialize_gradcam()
        
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # Save temporarily
#         temp_path = "temp.jpg"
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # Run YOLO + CNN
#         output_img, detections, gradcam_crops = self.detect_and_classify(temp_path)

#         # Create directories
#         os.makedirs("media/predictions", exist_ok=True)
#         os.makedirs("media/gradcam", exist_ok=True)
        
#         # Clear old files
#         for old_file in glob.glob("media/predictions/*.jpg"):
#             os.remove(old_file)
#         for old_file in glob.glob("media/gradcam/*.jpg"):
#             os.remove(old_file)

#         # Generate unique filename
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         unique_id = uuid.uuid4().hex[:6]

#         # Save annotated image
#         annotated_filename = f"annotated_{timestamp}_{unique_id}.jpg"
#         annotated_path = os.path.join("media", "predictions", annotated_filename)
#         cv2.imwrite(annotated_path, output_img)

#         # Save Grad-CAM images
#         gradcam_urls = []
#         for grad_item in gradcam_crops:
#             gradcam_filename = f"gradcam_{timestamp}_{unique_id}_fish{grad_item['index']}_{grad_item['label']}.jpg"
#             gradcam_path = os.path.join("media", "gradcam", gradcam_filename)
#             cv2.imwrite(gradcam_path, grad_item['image'])
            
#             # Build URL
#             gradcam_url = request.build_absolute_uri(f"/media/gradcam/{gradcam_filename}")
#             gradcam_urls.append({
#                 "fish_index": grad_item['index'],
#                 "label": grad_item['label'],
#                 "gradcam_url": gradcam_url
#             })

#         # Build annotated image URL
#         image_url = request.build_absolute_uri(f"/media/predictions/{annotated_filename}")

#         return Response({
#             "image_url": image_url,
#             "detections": detections,
#             "gradcam_visualizations": gradcam_urls
#         })

# import os
# import glob
# import uuid
# from datetime import datetime

# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torchvision import models
# from ultralytics import YOLO

# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.authentication import TokenAuthentication
# from rest_framework.response import Response


# class GradCAMResNet:
#     """Grad-CAM for PyTorch ResNet models"""
    
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
        
#         # Register hooks
#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_full_backward_hook(self.save_gradient)
    
#     def save_activation(self, module, input, output):
#         self.activations = output.detach()
    
#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
    
#     def generate_cam(self, input_tensor, target_class):
#         """
#         Generate Grad-CAM heatmap
#         input_tensor: preprocessed image tensor (1, 3, 224, 224)
#         target_class: target class index
#         """
#         # Forward pass
#         self.model.eval()
#         output = self.model(input_tensor)
        
#         # Backward pass
#         self.model.zero_grad()
#         output[0, target_class].backward()
        
#         # Get gradients and activations
#         gradients = self.gradients[0]  # [C, H, W]
#         activations = self.activations[0]  # [C, H, W]
        
#         # Calculate weights (global average pooling of gradients)
#         weights = gradients.mean(dim=(1, 2))  # [C]
        
#         # Weighted combination of activation maps
#         cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
#         for i, w in enumerate(weights):
#             cam += w * activations[i]
        
#         # Apply ReLU (only positive influence)
#         cam = F.relu(cam)
        
#         # Normalize to 0-1
#         if cam.max() > 0:
#             cam = cam - cam.min()
#             cam = cam / cam.max()
        
#         return cam.cpu().numpy()
    
#     def apply_heatmap(self, original_img, heatmap):
#         """
#         Overlay heatmap on original image
#         original_img: original crop image (H, W, 3) in BGR
#         heatmap: normalized heatmap (H', W')
#         """
#         # Resize heatmap to match image size
#         h, w = original_img.shape[:2]
#         heatmap_resized = cv2.resize(heatmap, (w, h))
        
#         # Convert to RGB colormap
#         heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
#         # Overlay: 60% original + 40% heatmap
#         overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
        
#         return overlay


# class ScanView(APIView):
#     authentication_classes = [TokenAuthentication]
#     permission_classes = [IsAuthenticated]
#     parser_classes = [MultiPartParser, FormParser]

#     # =======================
#     # Load Models Once
#     # =======================
#     yolo_model = YOLO("models/yolo/yolov8-1.pt")

#     # Load CNN model with checkpoint format
#     checkpoint = torch.load("models/cnn/resnet18-4.pth", map_location="cpu")
#     cnn_model = models.resnet18(pretrained=False)
#     cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
    
#     # Handle both old and new checkpoint formats
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         # New format (from your training script)
#         cnn_model.load_state_dict(checkpoint['model_state_dict'])
#         dryness_labels = checkpoint.get('classes', ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET'])
#     else:
#         # Old format (direct state dict)
#         cnn_model.load_state_dict(checkpoint)
#         dryness_labels = ['ALMOST_DRY', 'DRY', 'PARTIALLY_DRY', 'WET']
    
#     cnn_model.eval()

#     # Initialize Grad-CAM with ResNet18's last convolutional layer
#     gradcam = GradCAMResNet(cnn_model, cnn_model.layer4[-1])

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # =======================
#     # Detect + Classify
#     # =======================
#     def detect_and_classify(self, image_path):
#         image = cv2.imread(image_path)
#         results = self.yolo_model.predict(source=image_path, conf=0.5, verbose=False)[0]

#         class_names = results.names
#         detections = []
#         gradcam_crops = []

#         colors = {
#             "DRY": (79, 255, 79),              # green
#             "ALMOST_DRY": (77, 246, 255),      # yellow
#             "PARTIALLY_DRY": (71, 190, 255),   # orange
#             "WET": (255, 175, 79),             # blue
#             "REJECT": (41, 41, 255)            # red
#         }

#         for idx, box in enumerate(results.boxes):
#             cls_id = int(box.cls[0])
#             cls_name = class_names[cls_id]
#             conf = float(box.conf[0])

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = image[y1:y2, x1:x2]

#             if crop.size == 0:
#                 continue

#             # 🐟 Classify fish dryness
#             if cls_name.lower() == "fish":
#                 input_tensor = self.transform(crop).unsqueeze(0)
#                 with torch.no_grad():
#                     outputs = self.cnn_model(input_tensor)
#                     _, predicted = torch.max(outputs, 1)
#                     predicted_class = predicted.item()
#                     label = self.dryness_labels[predicted_class]
#                     color = colors.get(label, (255, 255, 255))
                
#                 # Generate Grad-CAM visualization
#                 try:
#                     # Need to enable gradients for Grad-CAM
#                     input_tensor.requires_grad = True
#                     heatmap = self.gradcam.generate_cam(input_tensor, predicted_class)
#                     gradcam_overlay = self.gradcam.apply_heatmap(crop.copy(), heatmap)
                    
#                     gradcam_crops.append({
#                         "index": idx,
#                         "image": gradcam_overlay,
#                         "label": label
#                     })
#                     print(f"✓ Grad-CAM generated for fish {idx}: {label}")
#                 except Exception as e:
#                     print(f"✗ Warning: Grad-CAM generation failed for fish {idx}: {e}")

#             # 🚫 Reject fish
#             elif cls_name.lower() == "reject":
#                 label = "REJECT"
#                 color = colors["REJECT"]

#             detections.append({
#                 "class": cls_name,
#                 "label": label,
#                 "confidence": round(conf, 2),
#                 "box": [x1, y1, x2, y2]
#             })

#             # Draw box
#             box_thickness = 5
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

#             # Label
#             text_scale = 1.5
#             text_thickness = 3
#             font = cv2.FONT_HERSHEY_SIMPLEX

#             (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
#             label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
#             label_bg_bottom_right = (x1 + text_width + 10, y1)

#             cv2.rectangle(image, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
#             cv2.putText(image, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

#         return image, detections, gradcam_crops

#     # =======================
#     # POST (Upload Image)
#     # =======================
#     def post(self, request):
#         image = request.FILES.get('image')
#         if not image:
#             return Response({"detail": "No image uploaded"}, status=400)

#         # ✅ Save uploaded file temporarily
#         temp_path = "temp.jpg"
#         with open(temp_path, "wb+") as f:
#             for chunk in image.chunks():
#                 f.write(chunk)

#         # ✅ Run detection
#         output, detections, gradcam_crops = self.detect_and_classify(temp_path)

#         # ✅ Make sure directories exist
#         os.makedirs("media/predictions", exist_ok=True)
#         os.makedirs("media/gradcam", exist_ok=True)

#         # ✅ Delete previous predictions (optional)
#         for old_file in glob.glob("media/predictions/*.jpg"):
#             os.remove(old_file)
#         for old_file in glob.glob("media/gradcam/*.jpg"):
#             os.remove(old_file)

#         # ✅ Create timestamp for filenames
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         unique_id = uuid.uuid4().hex[:6]

#         # ✅ Save the annotated image
#         annotated_filename = f"annotated_{timestamp}_{unique_id}.jpg"
#         annotated_path = os.path.join("media", "predictions", annotated_filename)
#         cv2.imwrite(annotated_path, output)

#         # ✅ Save Grad-CAM images
#         gradcam_urls = []
#         for grad_item in gradcam_crops:
#             gradcam_filename = f"gradcam_{timestamp}_{unique_id}_fish{grad_item['index']}_{grad_item['label']}.jpg"
#             gradcam_path = os.path.join("media", "gradcam", gradcam_filename)
#             cv2.imwrite(gradcam_path, grad_item['image'])
            
#             # Build URL
#             gradcam_url = request.build_absolute_uri(f"/media/gradcam/{gradcam_filename}")
#             gradcam_urls.append({
#                 "fish_index": grad_item['index'],
#                 "label": grad_item['label'],
#                 "gradcam_url": gradcam_url
#             })

#         # ✅ Convert path to URL-friendly format (use forward slashes)
#         relative_path = annotated_path.replace("\\", "/")

#         # ✅ Build correct absolute URL for response
#         image_url = request.build_absolute_uri("/" + relative_path.replace("media/", "media/"))

#         return Response({
#             "image_url": image_url,
#             "detections": detections,
#             "gradcam_visualizations": gradcam_urls
#         })
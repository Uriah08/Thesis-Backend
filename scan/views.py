from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import TokenAuthentication

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
import cv2
import os
import numpy as np
from tempfile import NamedTemporaryFile

import uuid
from datetime import datetime
import glob

class ScanView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    # =======================
    # Load Models Once
    # =======================
    yolo_model = YOLO("models/yolo/yolov8-1.pt")

    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
    cnn_model.load_state_dict(torch.load("models/cnn/resnet18-1.pth", map_location="cpu"))
    cnn_model.eval()

    dryness_labels = ['FULLY_DRY', 'ALMOST_DRY', 'PARTIALLY_DRY', 'WET']

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # =======================
    # Detect + Classify
    # =======================
    def detect_and_classify(self, image_path):
        image = cv2.imread(image_path)
        results = self.yolo_model.predict(source=image_path, conf=0.5, verbose=False)[0]

        class_names = results.names
        detections = []

        colors = {
            "FULLY_DRY": (79, 255, 79),        # green
            "ALMOST_DRY": (77, 246, 255),     # yellow
            "PARTIALLY_DRY": (71, 190, 255),  # orange
            "WET": (255, 175, 79),              # blue
            "REJECT": (41, 41, 255)           # red
        }

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # 🐟 Classify fish dryness
            if cls_name.lower() == "fish":
                input_tensor = self.transform(crop).unsqueeze(0)
                with torch.no_grad():
                    outputs = self.cnn_model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    label = self.dryness_labels[predicted.item()]
                    color = colors[label]

            # 🚫 Reject fish
            elif cls_name.lower() == "reject":
                label = "REJECT"
                color = colors["REJECT"]

            detections.append({
                "class": cls_name,
                "label": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

            # Draw box
            box_thickness = 5
            cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

            # Label
            text_scale = 1.5
            text_thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_width, text_height), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
            label_bg_top_left = (x1, max(y1 - text_height - 10, 0))
            label_bg_bottom_right = (x1 + text_width + 10, y1)

            cv2.rectangle(image, label_bg_top_left, label_bg_bottom_right, color, cv2.FILLED)
            cv2.putText(image, label, (x1 + 5, y1 - 5), font, text_scale, (0, 0, 0), text_thickness, lineType=cv2.LINE_AA)

        return image, detections

    # =======================
    # POST (Upload Image)
    # =======================
    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return Response({"detail": "No image uploaded"}, status=400)

        # ✅ Save uploaded file temporarily
        temp_path = "temp.jpg"
        with open(temp_path, "wb+") as f:
            for chunk in image.chunks():
                f.write(chunk)

        # ✅ Run detection
        output, detections = self.detect_and_classify(temp_path)
        print(detections)

        # ✅ Make sure predictions directory exists
        os.makedirs("media/predictions", exist_ok=True)

        # ✅ Delete previous predictions (optional)
        for old_file in glob.glob("media/predictions/*.jpg"):
            os.remove(old_file)

        # ✅ Create new annotated filename
        filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        output_path = os.path.join("media", "predictions", filename)

        # ✅ Save the annotated image
        cv2.imwrite(output_path, output)

        # ✅ Convert path to URL-friendly format (use forward slashes)
        relative_path = output_path.replace("\\", "/")

        # ✅ Build correct absolute URL for response
        image_url = request.build_absolute_uri("/" + relative_path)

        return Response({
            "image_url": image_url,
            "detections": detections
        })


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

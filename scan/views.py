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


class ScanView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    # =======================
    # Load Models Once
    # =======================
    yolo_model = YOLO("models/training/weights/best.pt")

    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 4)
    cnn_model.load_state_dict(torch.load("models/dryness_cnn.pth", map_location="cpu"))
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
            "FULLY_DRY": (0, 255, 0),        # green
            "ALMOST_DRY": (0, 255, 255),     # yellow
            "PARTIALLY_DRY": (0, 165, 255),  # orange
            "WET": (255, 0, 0),              # blue
            "REJECT": (0, 0, 255)            # red
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

        # ✅ Save annotated image
        output_path = "media/predictions/annotated.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output)

        # ✅ Return image URL and detections
        return Response({
            "image_url": request.build_absolute_uri("/" + output_path),
            "detections": detections
        })

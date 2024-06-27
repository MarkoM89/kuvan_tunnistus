from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

source = 'rtsp://192.168.0.12:8554/mjpeg/1'

# Use the model
results = model.predict(source, show = True)

print(results)
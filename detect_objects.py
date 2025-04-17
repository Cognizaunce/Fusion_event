from ultralytics import YOLO
import cv2  # OpenCV for image processing
import numpy
from matplotlib import pyplot as plt

# Load a model
model = YOLO("models/yolo11l.pt")  # load a custom model

# Predict with the model
results = model("data\CameraA\A_001.png")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes .xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box

# Save the result image with predictions
for result in results:
    img_with_boxes = result.plot()  # Annotated image as a NumPy array
    save_path = "prediction.png"
    
    import cv2
    cv2.imwrite(save_path, img_with_boxes)
    print(f"Saved prediction to {save_path}")
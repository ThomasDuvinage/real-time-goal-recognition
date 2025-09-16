import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 from ultralytics


class pluginZed:

    # Initialize the ZED camera
    def __init__(self, model_path,open_camera=True):
        self.zed = sl.Camera()
        if open_camera:
            #init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, camera_fps=30)
            init_params = sl.InitParameters()
            init_params.set_from_stream("150.18.224.24",30000)
            init_params.sdk_verbose = 1
            self.zed.open(init_params)
        self.model = YOLO(model_path)  # Load YOLOv8 model
        self.classes = list(self.model.names.values()) 
        self.eye_tracker = None
        
    def draw_gaze(self,frame,gaze_position,size=12, color=(0,255,255), thickness=2):
    	x,y = map(int,gaze_position)
    	cv2.line(frame, (x-size,y), (x+size,y), color, thickness)
    	cv2.line(frame, (x,y-size), (x,y+size), color, thickness)
	


    # Perform YOLOv8 object detection
    def detect_objects(self, frame, prediction_conf_threshold=0.5):
        ##results = self.model(frame)  # Inference on the frame
        results = self.model.predict(source=frame, conf=prediction_conf_threshold, verbose=False)[0]

        detections = []
        for result in results:
            # Parse detection results
            boxes = result.boxes  # Get bounding boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])  # Class index
                prediction_confidence = float(box.conf[0])  # Confidence
                if prediction_confidence > prediction_conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of bounding box
                    mask = result.masks.data[i].cpu().numpy() if result.masks is not None else None
                    #print(f"mask initial: {mask}")
                    detections.append({
                        "class_id": class_id,
                        "confidence": prediction_confidence,
                        "box": [x1, y1, x2 - x1, y2 - y1],  # Format as [x, y, w, h]
                        "mask": mask
                    })
        return detections

    # Draw bounding boxes on the frame
    def draw_bounding_boxes(self, frame, detections, classes, labels=None, confidences=None):
        for detection in detections:
            x, y, w, h = detection["box"]
            class_id = detection["class_id"]
            confidence = detection["confidence"]

            # If custom labels are provided, use them; otherwise, use the default class names
            label = labels[class_id] if labels else f"{classes[class_id]}"
            
            # If custom confidences are provided, use them; otherwise, use the detected confidence
            display_confidence = confidences[class_id] if confidences else confidence
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)        # TO DO: Change the color according to goal prediction value
            cv2.putText(frame, f"{label}: {display_confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw most likely goal for each object
    def draw_goal_boxes(self, frame, object_boxes, object_goals):
        for obj in object_boxes:
            x, y, w, h = object_boxes[obj]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            ##print(f"obj: {obj}, object_goals[obj]: {object_goals[obj]}")
            if obj not in object_goals or object_goals[obj][0] is None:
                label = f"analyzing({obj}) (?)"
                display_confidence = 0.0
            else:
                label = object_goals[obj][0]
                display_confidence = object_goals[obj][1]
                    # TO DO: Change the color according to goal prediction value
            cv2.putText(frame, f"{label}: {display_confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            
            

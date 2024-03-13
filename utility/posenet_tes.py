from ultralytics import YOLO
import cv2
import socket
import cv2
import pickle
import struct
import torch
import numpy as np
from ultralytics import YOLO
import re
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
import cvzone

def track_detect (detections, img, tracker):
    tracks = tracker.update_tracks(detections, frame=img)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        x1, y1, x2, y2= bbox
        x1, y1, x2,y2 = int (x1), int (y1), int (x2), int (y2)
        w,h = x2-x1, y2-y1
        cvzone.putTextRect(img, f'ID: {(track_id)}', (x1,y1), scale=1, thickness=1, colorR= (0, 0, 255))
        cvzone.cornerRect(img, (x1, y1, w, h), 1-9, rt=1, colorR= (255, 0, 255))
    return img

def plot_boxes(results, img) :
    detections = []
    for r in results:
        boxes = r. boxes
        for box in boxes:
            x1, y1, x2, y2 = box. xyxy[0]
            x1, y1, x2, y2 = int (x1), int (y1), int (x2), int (y2) 
            w,h = x2-x1, y2-y1
            # Classname
            currentClass = "Person"
            # Confodence score
            conf = math.ceil(box.conf [0]*100) /100
            if conf > 0.5:
                detections.append((([x1, y1, w, h]), conf, currentClass))
    return detections, img

model = YOLO('yolov8x-pose.pt')  
tracker=DeepSort()

cap = cv2.VideoCapture(1)

while True:
    ret,frame=cap.read()
    # Process frame with YOLOv5
    results = model(frame)
    # Display results
    
    #Locate users

    #annotated_frame = results[0].plot()
    detections, frame=plot_boxes(results,frame)
    detect_frame=track_detect(detections,frame,tracker)
    cv2.imshow("Received", detect_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




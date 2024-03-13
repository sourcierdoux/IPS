import socket
import cv2
import pickle
import struct
import torch
import numpy as np
from ultralytics import YOLO
import glob
import re

image_folder=glob.glob('reference_pics6/*.jpg')

model = YOLO('yolov8n.pt')
coordinate_mapping={}




model.predict(source="0", show=True, conf=0.4, stream=True, classes=0, hide_conf=True)
for image in image_folder:
    middle_x,middle_y=0,0
    frame = cv2.imread(image)
    results = model(frame)
    bboxes = results[0].boxes.xyxy.tolist()
    
    if len(bboxes)==0:
        print('No subject detected for image '+image)
        continue
    elif len(bboxes)==1:
        if isinstance(bboxes[0], list):
        # It's a nested list, so access the first inner list
            inner_list = bboxes[0]
        else:
        # It's not a nested list, use the list directly
            inner_list = bboxes
        x1,y1,x2,y2=inner_list
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        # Convert from center format to top-left format
        
        # You can now use (x, y, w, h) as the coordinates of the bounding box
        middle_top_x,middle_top_y=int((x1+x2)/2),y1
        middle_bottom_x,middle_bottom_y=int((x1+x2)/2),y2
        #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)
        
        print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, middle_x={middle_x}, middle_y={middle_y}")
        annotated_frame = results[0].plot()
        cv2.imshow("Received", annotated_frame)
        cv2.waitKey(0)
    else:
        
        for i,bbox in enumerate(bboxes):
            x1,y1,x2,y2=bbox
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)
            # Convert from center format to top-left format
            
            # You can now use (x, y, w, h) as the coordinates of the bounding box
            middle_top_x,middle_top_y=int((x1+x2)/2),y1
            middle_bottom_x,middle_bottom_y=(int(x1+x2))/2,y2
            #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)
            print(f"\nBounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}, middle_x={middle_x}, middle_y={middle_y}")
            cv2.putText(frame, str(i), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        annotated_frame = results[0].plot()
        cv2.imshow("Received", annotated_frame)
        cv2.waitKey(0)
        n=input("\nWhich box to choose ?")
        if n=='non':
            continue
        cv2.destroyAllWindows()
        x1,y1,x2,y2=bboxes[int(n)]
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        middle_top_x,middle_top_y=int((x1+x2)/2),y1
        middle_bottom_x,middle_bottom_y=int((x1+x2)/2),y2

    #pattern = re.compile(r'\((\d+),(\d+)\)')
    pattern = re.compile(r'\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)')
    match = pattern.search(image)
    if match:
        # Convert x and y to integers and store as a tuple
        x, y = map(float, match.groups())
    coordinate_mapping[(middle_top_x,middle_top_y,middle_bottom_x,middle_bottom_y)]=(x,y)

file_name = "coordinate_mapping_undistorted.txt"

# Writing the dictionary to a file
with open(file_name, "w") as file:
    for key, value in coordinate_mapping.items():
        file.write(f"{key}: {value}\n")

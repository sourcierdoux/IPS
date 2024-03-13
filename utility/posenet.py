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
from tensorflow.keras.models import load_model

model = YOLO('yolov8x-pose')  
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.137.248'  # Replace with your Raspberry Pi's IP
port = 12345

client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")


def is_sitting(keypoints, bboxes,frame):
    for i,keypoint in enumerate(keypoints):
        string=''
        shoulder_right=keypoint[6]
        shoulder_left=keypoint[5]
        hips_right=keypoint[12]
        hips_left=keypoint[11]
        knee_left=keypoint[13]
        knee_right=keypoint[14]
        foot_left=keypoint[15]
        foot_right=keypoint[16]
        angle_left=calculate_angle(shoulder_left,hips_left,knee_left)
        print(angle_left)
        if angle_left>120:
            string='Standing'
        else:
            string='Sitting'
        x1,y1,x2,y2=bboxes[i]
        cv2.putText(frame, string, (int((x1+x2)/2), int(y2)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def calculate_angle(P1, P2, P3):
    # Create vectors
    A = np.array(P1) - np.array(P2)
    B = np.array(P3) - np.array(P2)

    # Calculate dot product
    dot_product = np.dot(A, B)

    # Calculate magnitudes
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude_A * magnitude_B)

    # Avoid numerical instability
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians, and then convert to degrees
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)

    return angle_degrees


while True:
    while len(data) < payload_size:
        packet = client_socket.recv(420*640)  # 4K
        if not packet: break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize frame
    frame = pickle.loads(frame_data)
    # Process frame with YOLOv5
    results = model(frame)
    
    # Display results
    
    #Locate users
    bboxes = results[0].boxes.xyxy.tolist()
    keypoints=results[0].keypoints.xy.tolist()
    is_sitting(keypoints,bboxes,frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Received", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

client_socket.close()


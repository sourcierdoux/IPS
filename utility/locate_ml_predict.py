import socket
import cv2
import pickle
import struct
import torch
import numpy as np
from ultralytics import YOLO
import re

from tensorflow.keras.models import load_model



H=np.array([[-6.79628864e+00, -5.71232480e+00,  8.79272192e+02],
 [-8.88748261e+00, -1.95484683e+01,  1.65251928e+03],
 [-4.79102185e-01, -8.53511667e-01,  1.00000000e+00]])

def apply_homography(coord, H):
    
    homogeneous_coord = np.append(coord,1)
    transformed = np.dot(H, homogeneous_coord)
    transformed /= transformed[2]
    
        
    return transformed[:2]

# Load the model
prediction_model = load_model('prediction.h5')

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov8', 'yolov8s', pretrained=True)
# model.classes = [0]  # Specify the class (people)
# model.conf = 0.4

file_name = "coordinate_mapping.txt"

# Initialize an empty dictionary

# Read the file and construct the dictionary

model = YOLO('yolov8n.pt')
model.predict(source="0", show=True, conf=0.4, stream=True, classes=0, hide_conf=True)  # [0, 3, 5] for multiple classes


# Create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.137.248'  # Replace with your Raspberry Pi's IP
port = 12345

client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")

while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)  # 4K
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
    for i,bbox in enumerate(bboxes):
            x1,y1,x2,y2=bbox
            x1=int(x1)
            y1=int(y1)
            x2=int(x2)
            y2=int(y2)
            # Convert from center format to top-left format
            
            # You can now use (x, y, w, h) as the coordinates of the bounding box
            middle_top_x,middle_top_y, middle_bottom_x, middle_bottom_y=int((x1+x2)/2),y1,int((x1+x2)/2),y2
            #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)
            # Example image coordinate (u, v)
            top_array=apply_homography(np.array([middle_top_x,middle_top_y]),H)
            bottom_array=apply_homography(np.array([middle_bottom_x, middle_bottom_y]),H)
            print(top_array)
            print(bottom_array)
            X_test=np.concatenate([top_array, bottom_array],axis=0).reshape(1,4)
            print(X_test.shape[1])
            Y_test=prediction_model.predict(X_test)
            # Transform the point

            # Convert to non-homogeneous coordinates
            #print(Y_test)
            cv2.putText(frame, "("+str(np.around(Y_test[0][0],1))+","+str(np.around(Y_test[0][1],1))+")", (x2+10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    annotated_frame = results[0].plot()
    cv2.imshow("Received", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

client_socket.close()


    
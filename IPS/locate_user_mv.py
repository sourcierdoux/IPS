import socket
import cv2
import pickle
import struct
import time
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor
import threading
from locate_user_ble import *
import pandas as pd
from constants import FAN_POSITION
from beacon import Person

camera_position = (2.5, 15.5)
room_width = 12
room_length = 17
ceiling_height=1.75
FAN_POSITION=(7,14.5)
#DEFINE CONSTANTS
YOLO_VERBOSE=False

#Necessary Homography matrices
H_rest=np.array([[-6.79628864e+00, -5.71232480e+00,  8.79272192e+02],
 [-8.88748261e+00, -1.95484683e+01,  1.65251928e+03],
 [-4.79102185e-01, -8.53511667e-01,  1.00000000e+00]])

H2=np.array([[-8.97705994e-02, -5.48564876e-02,  8.97203259e+00],
 [-1.87609968e-01, -4.59289061e-01,  4.22488223e+01],
 [-1.01216183e-02, -2.36059501e-02,  1.00000000e+00]])

H_bottomright=np.array([[-1.81821162e-02,  1.07643382e-01,  5.80250925e+00],
 [-4.28251447e-02,  2.47430220e-01,  1.13334216e+01],
 [-3.33382707e-03,  1.68958200e-02,  1.00000000e+00]])

H_sitting = np.array([[-5.81701069e-02, -2.56460746e-02,  3.87479775e+00],
 [-1.38103624e-01, -3.78972332e-01,  3.96698322e+01],
 [-6.67663748e-03, -2.29830018e-02,  1.00000000e+00]])
H_bottomright_sitting=np.array([[-5.25308922e-03,  9.12906531e-03,  4.38034244e+00],
 [-2.83994998e-02,  6.16917281e-02,  1.19202244e+01],
 [-2.12417863e-03,  3.47914388e-03,  1.00000000e+00]])


# Import the right YOLO model
model = YOLO('yolov8m-pose.pt')  

# Connection to RaspberryPi
def camera_connect():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.137.149'  # Raspberry Pi IP
    port = 12345
    client_socket.connect((host_ip, port))
    data = b""
    payload_size = struct.calcsize("Q")
    return client_socket, data, payload_size

#Adjust coordinates in the camera 
def transform_coords(X,Y,camera_position):
    return X-camera_position[0],camera_position[1]-Y


def is_sitting_single(keypoint):
    #Determine if a user is sitting based on bodyparts keypoints

    # Access the interesting body parts
    shoulder_right=keypoint[6]
    shoulder_left=keypoint[5]
    hips_right=keypoint[12]
    hips_left=keypoint[11]
    knee_left=keypoint[13]
    knee_right=keypoint[14]
    foot_left=keypoint[15]
    foot_right=keypoint[16]

    # Define the angles for posture estimation
    angle_hips_left=calculate_angle(shoulder_left,hips_left,knee_left)
    angle_hips_right=calculate_angle(shoulder_right,hips_right,knee_right)
    angle_knee_left=calculate_angle(hips_left,knee_left,foot_left)
    angle_knee_right=calculate_angle(hips_right,knee_right,foot_right)

    #If the person is in profile view, only one side of its body can be seen
    if angle_hips_left>160 or angle_hips_right>160: 
        if angle_knee_right>160 or angle_knee_left>160:
            return False
    else:
        if (shoulder_right[0]>200 and shoulder_right[1]>40 and shoulder_right[0]<355 and shoulder_right[1]<60):
            return False
        return True

def calculate_angle(P1, P2, P3):
    #Function to calculate angle between three points

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


def apply_homography(coord, H):
    # Function to transform image coordinates to real world coordinates using homography matrix H
    
    homogeneous_coord = np.append(coord,1)
    transformed = np.dot(H, homogeneous_coord)
    transformed /= transformed[2]
    
    return transformed[:2]


def choose_h(coord_x,coord_y,sitting:bool):
    # Function to choose the right homography matrix

    if coord_x<640 and coord_x>325 and coord_y<(200/315)*(coord_x-325)+120:
        # We need to consider some parts of the image differently due to distortion
        # Consider the specific case of user on bottom right part of image
        if sitting==False:
            H=H_bottomright
        else:
            H=H_bottomright_sitting
    else:
        if sitting==False:
            H=H2
        else:
            H=H_sitting
    return H

def locate_from_homography(bboxes,frame,keypoints,ids,verbose=False):
    positions=[]
    persons=[]
    if (len(ids)==0):
        return frame,persons
    if verbose:
        print("Found users:")
    for i,bbox in enumerate(bboxes): # Access all the bounding boxes for each user
        x1,y1,x2,y2=bbox
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2) # Round the coordinates so that we can select a pixel
        middle_x,middle_y=int((x1+x2)/2),y1 # Use top-center coordinate of box for reference
        #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)

        # Transform the point
        sitting_bool=is_sitting_single(keypoint=keypoints[i]) # Determine if this user is sitting
        
        X,Y=apply_homography(np.array([middle_x,middle_y]),choose_h(middle_x,middle_y,sitting=sitting_bool)) # Apply the correct homography depending on posture
        
        
        cv2.putText(frame, "("+str(np.around(X,1))+","+str(np.around(Y,1))+")", (x2+10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Display coordinates
        cv2.putText(frame, "Sitting" if sitting_bool else "Standing", (int((x1+x2)/2), int(y2)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Display posture
        
        if verbose:
            print(f"Id: {ids[i]}, position: {X,Y}")
        
        positions.append((X,Y))
        persons.append(Person(ids[i],x=X,y=Y))
        #persons.append((ids[i],X,Y))
    
    return frame,persons




def plot_positions(shared_data):
    """Main plot of BLE and MV positions simultaneously"""
    while True:
        if 'MV' in shared_data:
            if 'BLE' in shared_data:
                plot_room(shared_data['MV'],shared_data['BLE'])
            else:
                plot_room(shared_data['MV'])
        time.sleep(0.3)


def plot_room(persons,ble_position=None):
    #Plot the room and position of the marker.
    
    plt.clf()  # clear the figure

    # Plot the room as a rectangle
    plt.gca().add_patch(plt.Rectangle((0, 0), room_width, room_length, fill=None))#
    #Plot fan position
    plt.scatter(room_width-FAN_POSITION[0],FAN_POSITION[1], color='brown')
    plt.text(int(room_width-FAN_POSITION[0]), int(FAN_POSITION[1]), "FAN", fontsize=9, ha='right', va='bottom')
    
    #Set of colors used for representing different user ids
    colors = ['red', 'green', 'yellow', 'purple','black'] 
    # Plot the current position of the markers
    for person, color in zip(persons,colors):
        X=room_width-person.x
        Y=person.y
        plt.scatter(X,Y, color=color)
        plt.text(int(X), int(Y), person.id, fontsize=9, ha='right', va='bottom')

    #Plot BLE position
    if ble_position is not None:
        plt.scatter(room_width-ble_position[0],ble_position[1],color='blue')
        plt.text(int(room_width-ble_position[0]), int(ble_position[1]), "BLE", fontsize=9, ha='right', va='bottom')
    
    #plt.scatter(*position_square, color='green')

    plt.xlim(0, room_width)
    plt.ylim(0, room_length)
    plt.xticks(range(room_width + 1))
    plt.yticks(range(room_length + 1))
    plt.grid(which='both', color='black', linewidth=0.5)

    plt.draw()
    plt.pause(0.1)  # pause to allow the figure to update

def plot_boxes(results, img) :
    """Function to get bouding box infos"""
    detections = []
    for r in results:
        boxes = r. boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int (x1), int (y1), int (x2), int (y2) 
            w,h = x2-x1, y2-y1
            # Classname
            currentClass = "Person"
            # Confidence score
            
            detections.append((([x1, y1, w, h]), box.conf, currentClass))
    return detections, img

def d_points(x1,y1,x2,y2):
    #Calculation of distance between two points
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def closest_person(persons, position):
    #Returns the closest person to the BLE position
    return min(persons, key=lambda f: d_points(f.x,f.y,position[0],position[1]))


def fetch_image(client_socket, data, payload_size):
    #Function to fetch a frame from the raspberrypi camera
    while len(data) < payload_size:
        packet = client_socket.recv(480*640)  # Exact shape of the image from the camera
        if not packet: break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(480*640)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data) 
    return frame, data

def locate_now(frame,plot=True):
    #Single execution function to return all Person objects found in a frame

    #Running YOLOv8 model with tracking and pose
    results = model.track(frame, tracker="IPS/tracker_modified.yaml",persist=True, verbose=False)
    bboxes = results[0].boxes.xyxy.tolist() #Accessing the bounding boxes

    keypoints=results[0].keypoints.xy.tolist() #Accessing the pose estimated keypoints
    if results[0].boxes.id is not None:
        ids=results[0].boxes.id.int().cpu().tolist()
    else:
        ids=[]
    frame,persons=locate_from_homography(bboxes,frame,keypoints,ids) #run the localization functions
    annotated_frame = results[0].plot(kpt_radius=1,kpt_line=False) #Plot the bounding box and the pose
    if plot==True:
        cv2.imshow("Received", annotated_frame) #Open window to display
    
    return persons

def evaluate_mv_error(current_x,current_y):
    """Function to calculate the error of MV positioning"""
    client_socket, data, payload_size = camera_connect()
    dict={}
    for i in range(50):
        
        frame, data=fetch_image(client_socket=client_socket, data=data, payload_size=payload_size)  

        persons=locate_now(frame,plot=True)
        plot_room(persons)
        for person in persons:
            if person.id not in dict:
                dict[person.id] = []
            dict[person.id].append((person.x, person.y),(current_x, current_y))

    df = pd.DataFrame({key: pd.Series(value) for key, value in dict.items()})

    return df

def mv_positioning(shared_data, plot=False):
    """Main function to run MV positioning"""
    client_socket, data, payload_size = camera_connect()
    while True:
        frame, data=fetch_image(client_socket=client_socket, data=data, payload_size=payload_size)
        persons=locate_now(frame,plot=plot)
        if shared_data is not None:
            shared_data['MV']= persons
        if plot==True:
            plot_room(persons)
        if cv2.waitKey(1) == ord('q'): #Press Q to quit
            break
    client_socket.close()

# Connect to the Raspberry Pi server
def init_connect_fan():

    client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket2.connect(('192.168.137.32', 12346))  
    return client_socket2
# Send a message



def match_id(shared_data):
    #Get a match between BLE and MV positions
    matched= closest_person(shared_data['MV'],shared_data['BLE'])
    print(f'A match has been found with person {matched}')
    return matched


    
    


import cv2
import pickle
import struct
import torch
import numpy as np
from ultralytics import YOLO
import glob
#from tensorflow.keras.models import load_model
import re

image_name='/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/useless_stuff/sitting_pictures/(4,6).jpg'

model = YOLO('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/yolov8m-pose.pt')
coordinate_mapping={}

DIM=(640,480)
K=np.array([[312.21855677,  -0.58893547, 336.38175646],
 [  0.,         312.86966051, 247.45166147],
 [  0.,           0.,           1.        ]])
D=np.array([[-0.03977174],
 [ 0.02779442],
 [-0.01602481],
 [ 0.0038209 ]])

"""H_rest=np.array([[-6.79628864e+00, -5.71232480e+00,  8.79272192e+02],
 [-8.88748261e+00, -1.95484683e+01,  1.65251928e+03],
 [-4.79102185e-01, -8.53511667e-01,  1.00000000e+00]])"""

H2=np.array([[-8.97705994e-02, -5.48564876e-02,  8.97203259e+00],
 [-1.87609968e-01, -4.59289061e-01,  4.22488223e+01],
 [-1.01216183e-02, -2.36059501e-02,  1.00000000e+00]])

H_bottomright=np.array([[-1.81821162e-02,  1.07643382e-01,  5.80250925e+00],
 [-4.28251447e-02,  2.47430220e-01,  1.13334216e+01],
 [-3.33382707e-03,  1.68958200e-02,  1.00000000e+00]])

H_sitting=np.array([[-5.81701069e-02, -2.56460746e-02,  3.87479775e+00],
 [-1.38103624e-01, -3.78972332e-01,  3.96698322e+01],
 [-6.67663748e-03, -2.29830018e-02,  1.00000000e+00]])
H_bottomright_sitting=np.array([[-5.25308922e-03,  9.12906531e-03,  4.38034244e+00],
 [-2.83994998e-02,  6.16917281e-02,  1.19202244e+01],
 [-2.12417863e-03,  3.47914388e-03,  1.00000000e+00]])
# Read the file and construct the dictionary


def is_sitting_single(keypoint):
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
    #print(angle_left)
    print(angle_knee_left)
    print(angle_knee_right)
    if angle_hips_left>160 or angle_hips_right>160: 
        if angle_knee_right>160 or angle_knee_left>160:
            print("I am here")
            return False
        
    return True

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





def undistort(img,balance=0, dim2=None, dim3=None):
	dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
	assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
	if not dim2:
		dim2 = dim1
	if not dim3:
		dim3 = dim1
	scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
	scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
	# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
	new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
	undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
	return undistorted_img

def apply_homography(coord, H):
    
    homogeneous_coord = np.append(coord,1)
    transformed = np.dot(H, homogeneous_coord)
    transformed /= transformed[2]
    
        
    return transformed[:2]


def locate_from_homography(bboxes,frame,keypoints):
    for i,bbox in enumerate(bboxes): # Access all the bounding boxes for each user
        x1,y1,x2,y2=bbox
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2) # Round the coordinates so that we can select a pixel
        middle_x,middle_y=int((x1+x2)/2),y1 # Use top-center coordinate of box for reference
        #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)

        # Transform the point
        sitting_bool=is_sitting_single(keypoint=keypoints[i]) # Determine if this user is sitting
        X,Y=apply_homography(np.array([middle_x,middle_y]),choose_h(middle_x,middle_y,sitting=sitting_bool)) # Apply the correct homography depending on posture
        #cv2.putText(frame, "("+str(np.around(X,1))+","+str(np.around(Y,1))+")", (x2+10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Display coordinates
        #cv2.putText(frame, "Sitting" if sitting_bool else "Standing", (int((x1+x2)/2), int(y2)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Display posture
    return frame

def locate_ml(bboxes, frame):
    for i,bbox in enumerate(bboxes):
        x1,y1,x2,y2=bbox
        x1=int(x1)
        y1=int(y1)
        x2=int(x2)
        y2=int(y2)
        # Convert from center format to top-left format
        
        # You can now use (x, y, w, h) as the coordinates of the bounding box
        middle_top_x,middle_top_y, middle_bottom_x, middle_bottom_y=int((x1+x2)/2),y1,int((x1+x2)/2),y2
        cv2.circle(frame, (240, 250), radius=5, color=(0, 255, 0), thickness=-1)
        # Example image coordinate (u, v)
        top_array=apply_homography(np.array([middle_top_x,middle_top_y]),choose_h(middle_top_x,middle_top_y))
        bottom_array=apply_homography(np.array([middle_bottom_x, middle_bottom_y]),choose_h(middle_bottom_x,middle_bottom_y))
        X_test=np.concatenate([top_array, bottom_array],axis=0).reshape(1,4)
        Y_test=prediction_model.predict(X_test)
        # Transform the point

        # Convert to non-homogeneous coordinates
        #print(Y_test)
        cv2.putText(frame, "("+str(np.around(Y_test[0][0],1))+","+str(np.around(Y_test[0][1],1))+")", (x2+10, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

model.predict(source="0", show=True, conf=0.4, stream=True, classes=0, show_conf=False)
middle_x,middle_y=0,0
frame = cv2.imread(image_name)
results = model(frame)
bboxes = results[0].boxes.xyxy.tolist()
if len(bboxes)==0:
    print('No subject detected for image '+image_name)
elif len(bboxes)==1:
    x1,y1,x2,y2=bboxes[0]
    x1=int(x1)
    y1=int(y1)
    x2=int(x2)
    y2=int(y2)
    # Convert from center format to top-left format
    
    # You can now use (x, y, w, h) as the coordinates of the bounding box
    middle_x,middle_y=int((x1+x2)/2),y1
    #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.circle(frame, (310, 55), radius=10, color=(255, 255, 0), thickness=-1)
    print(f"Middle point: "+str(middle_x)+","+str(middle_y))
    cv2.putText(frame, "("+str(middle_x)+","+str(middle_y)+")", (x2+20, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("("+str(middle_x)+","+str(middle_y)+")")
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
        middle_bottom_x,middle_bottom_y=int((x1+x2)/2),y2
        #cv2.circle(frame, (middle_x, middle_y), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, (310, 55), radius=10, color=(255, 255, 0), thickness=-1)
        print(f"bounding box: {x1} {x2} {y1} {y2}")
        #cv2.putText(frame, str(i)+": ("+str(middle_top_x)+","+str(middle_top_y)+")", (x2-15, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("("+str(middle_top_x)+","+str(middle_top_y)+")")
keypoints=results[0].keypoints.xy.tolist()
frame=locate_from_homography(bboxes,frame, keypoints)
#frame=locate_ml(bboxes,frame)

annotated_frame = results[0][0].plot(labels=False,probs=False, kpt_line=False,kpt_radius=0)
cv2.imwrite('obstacles.jpg',annotated_frame)
cv2.imshow(image_name, annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
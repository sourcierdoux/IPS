import socket
import cv2
import pickle
import struct
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)
undistort_image=False



DIM=(640,480)
 
K2=np.array([[305.79108402,  -0.34531888, 340.99123634],
 [  0.,         307.35449377, 246.94291485],
 [  0.,           0.,           1.        ]])
D2=np.array([[-0.07171792],
 [ 0.30986257],
 [-0.80833164],
 [ 0.66578076]])

K=np.array([[ 3.07560535e+02, -1.51167116e-01,  3.41731432e+02],
 [ 0.00000000e+00,  3.08542830e+02,  2.47659574e+02],
 [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
D=np.array([[-0.03448471],
 [ 0.01519671],
 [-0.02956966],
 [ 0.01412123]])


# Set up the socket for transmission
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.137.149'  # Replace with your Raspberry Pi's IP address
port = 12345
socket_address = (host_ip, port)

# Bind and listen
server_socket.bind(socket_address)
server_socket.listen(5)
print("Listening at:", socket_address)

# Accept a connection
client_socket, addr = server_socket.accept()
print('Got connection from', addr)




while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate and resize the frame
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    #frame = cv2.resize(frame, (640, 480))
    
    # Serialize the frame
    serialized_frame = pickle.dumps(frame)
    # Create a structure for sending data (message size + data)
    message = struct.pack("Q", len(serialized_frame)) + serialized_frame
    # Send the serialized frame
    client_socket.sendall(message)

cap.release()
client_socket.close()
server_socket.close()



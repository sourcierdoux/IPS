import socket
import cv2
import struct
import pickle

#initialize pose estimator

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.137.149'  # Replace with your Raspberry Pi's IP
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
    try:
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # convert to RGB
        
        # process the frame for pose detection
    
        
        # draw skeleton on the frame
        cv2.imshow("Received", frame)
    except:
        break
    
    if cv2.waitKey(1) == ord('q'):
        break

client_socket.close()
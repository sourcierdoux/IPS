import socket

# Connect to the Raspberry Pi server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.137.77', 12346))  # Replace with your Raspberry Pi's IP address

# Send a message
message = "Hello Raspberry Pi!"
client_socket.send(message.encode())

# Close the connection
client_socket.close()
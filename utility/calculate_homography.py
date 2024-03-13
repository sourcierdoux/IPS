import re
import numpy as np
import cv2

file_name = "homography_files/sitting_bottomright.txt"
loaded_coordinate_mapping = {}

# Read the file and construct the dictionary
with open(file_name, "r") as file:
    for line in file:
        # Extracting the key and value pairs using regular expression
        matches = re.findall(r'\((\d+), (\d+)\): \((\d+), (\d+)\)', line)
        if matches:
            key = tuple(map(int, matches[0][:2]))
            value = tuple(map(int, matches[0][2:]))
            loaded_coordinate_mapping[key] = value

imagePoints = []
realWorldPoints = []

for (x, y),(u, v)  in loaded_coordinate_mapping.items():
    imagePoints.append([u, v])
    realWorldPoints.append([x, y])

imagePoints = np.array(imagePoints, dtype=np.float32)
realWorldPoints = np.array(realWorldPoints, dtype=np.float32)

H, status = cv2.findHomography(imagePoints, realWorldPoints)

print(H)
print(status)
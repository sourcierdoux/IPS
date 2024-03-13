import tensorflow as tf
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

file_name = "train_XY.txt"
loaded_coordinate_mapping = {}

H=np.array([[-6.79628864e+00, -5.71232480e+00,  8.79272192e+02],
 [-8.88748261e+00, -1.95484683e+01,  1.65251928e+03],
 [-4.79102185e-01, -8.53511667e-01,  1.00000000e+00]])

H2=np.array([[-7.30468254e-01, -4.96631111e-01,  8.72283406e+01],
 [-1.04166130e+00, -2.21758116e+00,  2.13216357e+02],
 [-5.49137354e-02, -7.08366787e-02,  1.00000000e+00]])

def apply_homography(coords, H):
    transformed_coords = np.array([]).reshape(0,2)
    for coord in coords:
        homogeneous_coord = np.append(coord,1)
        transformed = np.dot(H, homogeneous_coord)
        transformed /= transformed[2]
        transformed_coords=np.append(transformed_coords,transformed[:2].reshape(1,2),axis=0)
        
    return transformed_coords

# Read the file and construct the dictionary
with open(file_name, "r") as file:
    for line in file:
        # Extracting the key and value pairs using regular expression
        #matches = re.findall(r'\((\d+), (\d+), (\d+), (\d+)\): \((\d+), (\d+)\)', line)
        matches = re.findall(r'\((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\): \((-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\)', line)


        if matches:
            key = tuple(map(float, matches[0][:4]))
            value = tuple(map(float, matches[0][4:]))
            loaded_coordinate_mapping[key] = value


top_points = []
bottom_points = []
realWorldPoints = []

for (x1, y1, x2, y2),(u, v)  in loaded_coordinate_mapping.items():
    top_points.append([x1, y1]) 
    bottom_points.append([x2, y2])
    realWorldPoints.append([u, v])


top_coordinates=apply_homography(np.array(top_points),H2)

bottom_coordinates=apply_homography(np.array(bottom_points),H2)
X_train=np.concatenate([top_coordinates, bottom_coordinates],axis=1)
print(X_train.shape[0])
Y_train=np.array(realWorldPoints)
#print(top_coordinates)
# Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)  # Output layer with 2 nodes for x and y coordinates
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, Y_train, epochs=200, batch_size=10, validation_split=0.2)

model.save('prediction_3.h5')
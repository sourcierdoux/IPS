import numpy as np
import bleak
import time
from bleak import BleakScanner
import asyncio
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

camera_position = (2.5, 15.5)
room_width = 12
room_length = 17
ceiling_height=1.75
FAN_POSITION=(7,14.5)

address_BLE2="CD3D8B09-2EAB-27E7-A67A-19263FE5369B"
address_BLE6="BE891D8E-0B63-C9EB-AF08-EFA63E30A1FD"
address_BLE10="33782DDB-6E6A-E7E6-1B45-45FFDC86CD72"
address_BLE3="4B94A02B-8F7A-0B61-FDBB-36F90C6AFABF"

address_BLE_device_1="47F437F8-0807-D40D-CDD7-1098351C398C"
address_BLE_device_2="76F34488-4ECB-C658-B1F4-56C1AC9D741F"
address_BLE_device_3="2DAF08FB-4088-5671-EF24-0D22646F1333"

iphone_address="7C67B5F1-A5C1-E3BB-AA12-7AC52CE8B672"
list_addresses=[address_BLE_device_1,address_BLE_device_2,address_BLE_device_3,address_BLE2,address_BLE6,address_BLE10,address_BLE3]


async def scan():
    distance_list=[]
    for address in list_addresses:
        device = await BleakScanner.find_device_by_address(address, timeout=5)
        distance_list.append(device.rssi)
    return np.array(distance_list)


def predict_position(model):
    list_positions=np.array([]).reshape(0,7)
    for i in range(10):
        current_measurement=asyncio.run(scan())
        while any(x>=0 for x in current_measurement):
            current_measurement=asyncio.run(scan())
        current_measurement=current_measurement.reshape(1,7)
        #print(current_measurement)
        list_positions= np.append(list_positions,current_measurement,axis=0)
    return np.mean(list_positions,axis=0).reshape(1,7)


def locate_fingerprint(shared_data,plot_final=False):
    with open('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/rf_model.pkl', 'rb') as file:
        model = pickle.load(file)

    array_rssi=predict_position(model)
    array_positions=model.predict(array_rssi)
    #array_positions_x=(labels%13)
    #array_positions_y=(labels//13)
    
    
    print(f"Predicted position {array_positions}")
    mean_pos=(array_positions[0][0],array_positions[0][1])
    #print(mean_squared_error((current_x*0.6,current_y*0.6),(mean_pos*0.6)))
    if shared_data is not None:
        shared_data['BLE']=(mean_pos[0],mean_pos[1])
    if plot_final:
        plot_final_finger(mean_pos)
    return mean_pos

    
def evaluate_fingerprint_error(file=None):
    current_x=input("enter current x:")
    current_y=input("enter current y:")
    position_array=np.array([]).reshape(0,4)
    for i in range(10):
        predicted_pos=locate_fingerprint(shared_data=None)
        row=np.array([predicted_pos[0],predicted_pos[1],current_x,current_y]).reshape(1,4)
        position_array=np.append(position_array,row,axis=0)

    if file == True:
        df=pd.read_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/errors_fingerprint.csv')
        values_df=pd.DataFrame(position_array,index=None,columns=['x_pred','y_pred', 'x_true','y_true'])
        df=pd.concat([df,values_df],ignore_index=True,axis=0)
        df.to_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/errors_fingerprint.csv',index=False)
    else:
        df=pd.DataFrame(position_array,index=None,columns=['x_pred','y_pred', 'x_true','y_true'])
        df.to_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/errors_fingerprint.csv',index=False)

def plot_final_finger(pos):
    plt.clf()  # clear the figure

    #Display fan position
    plt.gca().add_patch(plt.Rectangle((0, 0), room_width, room_length, fill=None))
    plt.scatter(room_width-FAN_POSITION[0],FAN_POSITION[1], color='brown')
    plt.text(int(room_width-FAN_POSITION[0]), int(FAN_POSITION[1]), "FAN", fontsize=9, ha='right', va='bottom')
    # Plot the room as a rectangle
    plt.scatter(room_width-camera_position[0],camera_position[1], color='black')
    plt.text(int(room_width-camera_position[0]), int(camera_position[1]), "Camera", fontsize=9, ha='right', va='bottom')

    
    # Plot the current position of the marker
    X=room_width-pos[0]
    Y=pos[1]   
    plt.scatter(X,Y, color="red")
    plt.text(int(X), int(Y), "BLE Predicted pos", fontsize=9, ha='right', va='bottom')

    
    #plt.scatter(*position_square, color='green')

    plt.xlim(0, room_width)
    plt.ylim(0, room_length)
    plt.xticks(range(room_width + 1))
    plt.yticks(range(room_length + 1))
    plt.grid(which='both', color='black', linewidth=0.5)

    plt.show()
    
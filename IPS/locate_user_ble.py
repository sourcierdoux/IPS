import asyncio
from bleak import BleakScanner
import time
import requests as re
import numpy as np
import math
import random

from signal_utils import *
from trilateration import *
from beacon import *
from plot import *
import pandas as pd






def locate_ble(shared_data,plot=False,n_measurement=10,current_position=False, final_plot=False):
    #Main function for localizing using BLE

    print("Starting to locate with BLE")

    #Define beacons parameters
    b1 = beacon(x=4*0.6,y=4.5*0.6,address="BE891D8E-0B63-C9EB-AF08-EFA63E30A1FD",name="BLE6",power_ref=-40)
    b2 = beacon(x=9.5*0.6,y=4.5*0.6,address="CD3D8B09-2EAB-27E7-A67A-19263FE5369B",name="BLE2",power_ref=-40)
    b3 = beacon(x=6.5*0.6,y=9.5*0.6,address="33782DDB-6E6A-E7E6-1B45-45FFDC86CD72",name="BLE10",power_ref=-40)
    b4 = beacon(x=10*0.6,y=15*0.6,address="4B94A02B-8F7A-0B61-FDBB-36F90C6AFABF",name="BLE3",power_ref=-37)

    ble1 = beacon(x=1.2*0.6,y=15.5*0.6,address="2DAF08FB-4088-5671-EF24-0D22646F1333",name="BLE_beacon_3",power_ref=-49)
    ble2= beacon(x=1.2*0.6,y=1.2*0.6,address="47F437F8-0807-D40D-CDD7-1098351C398C", name="BLE_beacon_1",power_ref=-49)
    ble3= beacon(x=11.5*0.6, y=9.1*0.6, address="76F34488-4ECB-C658-B1F4-56C1AC9D741F", name="BLE_beacon_2",power_ref=-49)

    #LOOP MEASUREMENTS AND CALCULATE POSITIONS
    signal, signal_no_filter, positions_list=loop_signal(iterations=n_measurement, list_beacons=[b1,b2,b3,b4,ble1,ble2,ble3], last_values=init_loop(list_beacons=[b1,b2,b3,b4,ble1,ble2,ble3]), current_position=current_position, plot=plot)
    #signal and signal_no_filter are for plotting signals after scan
    #plot_signal(signal=signal, signal_no_filter=signal_no_filter, n_measurement=n_measurement)
    
    avg_x,avg_y=np.ma.mean(np.ma.masked_equal(positions_list,0),axis=0)
    if shared_data is not None:
        shared_data['BLE']=(avg_x/0.6,avg_y/0.6)
    predicted_pos=(avg_x/0.6,avg_y/0.6)
    if final_plot:
        plot_final_trilat(predicted_pos)
    return predicted_pos


def evaluate_trilateration_error(file=None):
    """Function to calculate trilateration error based on two given coordinates x and y."""

    current_x=input("enter current x:")
    current_y=input("enter current y:")
    array_positions=np.array([]).reshape(0,4)
    for i in range(10):
        x,y=locate_ble(shared_data=None,plot=False,n_measurement=1)
        while (x,y)==(0,0):
            x,y=locate_ble(shared_data=None,plot=False,n_measurement=1)
        row=np.array([x,y,current_x,current_y]).reshape(1,4)
        array_positions=np.append(array_positions,row,axis=0)
    if file == True:
        df=pd.read_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/errors_trilateration.csv')
        values_df=pd.DataFrame(array_positions,index=None,columns=['x_pred','y_pred', 'x_true','y_true'])
        df=pd.concat([df,values_df],ignore_index=True,axis=0)
        df.to_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/errors_trilateration.csv',index=False)
    else:
        df=pd.DataFrame(array_positions,index=None,columns=['x_pred','y_pred', 'x_true','y_true'])
        df.to_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/errors_trilateration.csv',index=False)
    return pd.DataFrame(array_positions,index=None,columns=['x_pred','y_pred','x_true','y_true'])

def plot_final_trilat(pos):
    """Function used to plot the final trilaterated position"""

    camera_position = (2.5, 15.5)
    room_width = 12
    room_length = 17
    ceiling_height=1.75
    FAN_POSITION=(7,14.5)
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
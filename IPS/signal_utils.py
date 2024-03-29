import numpy as np
from beacon import *
import asyncio
from bleak import BleakScanner
import time
import requests as re
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from trilateration import *
from plot import plot_room_ble
from plot import *


N=2.4

def kalman_block(x, P, s, A, H, Q, R):

    """
    Prediction and update in Kalman filter

    input:
        - signal: signal to be filtered
        - x: previous mean state
        - P: previous variance state
        - s: current observation
        - A, H, Q, R: kalman filter parameters

    output:
        - x: mean state prediction
        - P: variance state prediction

    """

    # check laaraiedh2209 for further understand these equations

    x_mean = A * x + np.random.normal(0, Q, 1)
    P_mean = A * P * A + Q

    K = P_mean * H * (1 / (H * P_mean * H + R))
    x = x_mean + K * (s - H * x_mean)
    P = (1 - K * H) * P_mean

    return x, P


def kalman_filter(signal_value, last_value, A, H, Q, R):

    """

    Implementation of Kalman filter.
    Takes a signal and filter parameters and returns the filtered signal.

    input:
        - signal: signal to be filtered
        - A, H, Q, R: kalman filter parameters

    output:
        - filtered signal

    """


                                    # takes first value as first filter prediction
    P = 0                                         # set first covariance state value to zero

                # iterates on the entire signal, except the first element

    x, P = kalman_block(last_value, P, signal_value, A, H, Q, R)  # calculates next state prediction

    return x


def d_from_rssi(rssi, beacon):
    #Distance calculation
    distance = np.power(10,((beacon.power_ref - rssi)/(10*N)))
    return distance

def loop_signal(iterations: int, list_beacons: 'list[beacon]', last_values, current_position=False, plot=True):
    """Main function to loop scanning"""
    positions_list=np.array([]).reshape(0,2)
    n_beacons=len(list_beacons)
    x_current=0
    y_current=0
    if current_position==True:
        x_current=int(input("Enter current x: "))*0.6
        y_current=int(input("Enter current y: "))*0.6
    if plot==True:
        plt.ion()  # turn on interactive mode
        plt.figure()
    signal=np.array([]).reshape(0,n_beacons)
    signal_no_filter=np.array([]).reshape(0,n_beacons)
    instant_values=np.zeros((0,n_beacons))
    # Simulate moving the marker in the room
    for _ in range(iterations):  
        instant_values=run_discover_mean(list_beacons) #Fetch RSSI values
        signal_kalman_filter = kalman_filter(instant_values, last_values, A=1, H=1, Q=1e-3, R=0.01)

        signal_no_filter=np.append(signal_no_filter,instant_values.reshape(1,n_beacons),axis=0)
        last_values=signal_kalman_filter
        #print("\nFiltered RSSI:")
        #print(signal_kalman_filter)


        signal=np.append(signal,signal_kalman_filter.reshape(1,n_beacons),axis=0)
        for i,b in enumerate(list_beacons):
            b.d_to_user=d_from_rssi(signal_kalman_filter[i], b) 
        

        three_selected_beacons=select_three(list_beacons)
        run_hypot(three_selected_beacons)

        distances = np.array([b.d_2D for b in three_selected_beacons])
        x,y=select_case(three_selected_beacons[0],three_selected_beacons[1],three_selected_beacons[2])
        positions_list=np.append(positions_list,[(x,y)],axis=0)
        #x_square, y_square=locate_square(b1,b2,b3,distances)
        x_weight,y_weight=locate_weight(three_selected_beacons[0],three_selected_beacons[1],three_selected_beacons[2])

        if plot==True:
            plot_room_ble([x, y],[x_weight,y_weight],three_selected_beacons[0],three_selected_beacons[1],three_selected_beacons[2],x_current,y_current)
        time.sleep(0.1)  # pause for 1 second

    if plot==True:
        plt.ioff()  # turn off interactive mode
        plt.show()
    return signal, signal_no_filter, positions_list


async def discover(list_beacons):
    """Scanning function"""
    distance_list=[]
    for beacon in list_beacons:
        device = await BleakScanner.find_device_by_address(beacon.address,timeout=5)
        if device is not None:
            distance_list.append(device.rssi)
        else:
            distance_list.append(-100)
    return np.array(distance_list)


def run_discover_mean(list_beacons: 'list[beacon]'):
    """Function to perform moving average when scanning"""
    n_beacons=len(list_beacons)
    array_average=np.array([]).reshape(0,n_beacons)
    for i in range(0,2):
        value=np.zeros((1,n_beacons))
        while np.any(value >= 0):
            value=asyncio.run(discover(list_beacons)).reshape(1,n_beacons)
        array_average=np.append(array_average,value,axis=0)
        time.sleep(0.1)
    instant_values = np.mean(array_average,axis=0)
    return instant_values



def run_hypot(beacon_list: 'list[beacon]'):
    """Function to performn dimensional reduction """
    for b in beacon_list:
        b.d_2D=b.d_to_user
        """if ((b.d_to_user<ceiling_height)):
            b.d_2D=0.2
        else:
            b.d_2D=np.sqrt(b.d_to_user**2-ceiling_height**2)"""
    
  

def init_loop(list_beacons: 'list[beacon]'):
    """Function to obtain initial values for the Kalman filter"""
    n_beacons=len(list_beacons)
    signal=np.array([]).reshape(0,n_beacons)
    signal_no_filter=np.array([]).reshape(0,n_beacons)
    last_values=run_discover_mean(list_beacons=list_beacons)
    instant_values=np.zeros((0,n_beacons))
    # Simulate moving the marker in the room
    for _ in range(4):  # loop 4 times
        instant_values=run_discover_mean(list_beacons=list_beacons) #Fetch RSSI values
        signal_kalman_filter = kalman_filter(instant_values, last_values, A=1, H=1, Q=1e-3, R=0.01)

        signal_no_filter=np.append(signal_no_filter,instant_values.reshape(1,n_beacons),axis=0)
        last_values=signal_kalman_filter
        time.sleep(0.1)
    return last_values


def select_three(list_beacons: 'list[beacon]'):
    """Function to select the three closest beacons"""
    selected_beacons=sorted(list_beacons,key=lambda x: x.d_to_user)[:3]
    return selected_beacons
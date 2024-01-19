import asyncio
from bleak import BleakScanner
import time
import requests as re
import numpy as np
from scipy.optimize import least_squares
import math
import random

from signal_utils import *
from trilateration import *
from beacon import *
from plot import *




#---MAIN---

def locate_ble(shared_data,plot=False,n_measurement=10,current_position=False):
    
    #Define beacons
    b1 = beacon(x=3*0.6,y=4*0.6,address="BE891D8E-0B63-C9EB-AF08-EFA63E30A1FD",name="BLE6",power_ref=-40)
    b2 = beacon(x=9.5*0.6,y=4.5*0.6,address="CD3D8B09-2EAB-27E7-A67A-19263FE5369B",name="BLE2",power_ref=-40)
    b3 = beacon(x=9*0.6,y=6*0.6,address="33782DDB-6E6A-E7E6-1B45-45FFDC86CD72",name="BLE10",power_ref=-40)

    ble1 = beacon(x=1.5*0.6,y=15.5*0.6,address="2DAF08FB-4088-5671-EF24-0D22646F1333",name="BLE_beacon_3",power_ref=-48)
    ble2= beacon(x=1.2*0.6,y=1.2*0.6,address="47F437F8-0807-D40D-CDD7-1098351C398C", name="BLE_beacon_1",power_ref=-48)
    ble3= beacon(x=11*0.6, y=9.1*0.6, address="76F34488-4ECB-C658-B1F4-56C1AC9D741F", name="BLE_beacon_2",power_ref=-48)

    signal, signal_no_filter, positions_list=loop_signal(iterations=n_measurement, list_beacons=[ble1,ble2,ble3], last_values=init_loop(list_beacons=[ble1,ble2,ble3]), current_position=current_position, plot=plot)
    #plot_signal(signal=signal, signal_no_filter=signal_no_filter, n_measurement=n_measurement)
    avg_x,avg_y=np.ma.mean(np.ma.masked_equal(positions_list,0),axis=0)
    if shared_data is not None:
        shared_data['BLE']=(avg_x,avg_y)
    return avg_x/0.6,avg_y/0.6


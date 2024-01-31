import numpy as np
import bleak
import time
from bleak import BleakScanner
import asyncio
from tqdm import tqdm
import pandas as pd

current_x=input("enter current x:")
current_y=input("enter current y:")

address_BLE2="CD3D8B09-2EAB-27E7-A67A-19263FE5369B"
address_BLE6="BE891D8E-0B63-C9EB-AF08-EFA63E30A1FD"
address_BLE10="33782DDB-6E6A-E7E6-1B45-45FFDC86CD72"
address_BLE3="4B94A02B-8F7A-0B61-FDBB-36F90C6AFABF"

address_BLE_device_1="47F437F8-0807-D40D-CDD7-1098351C398C"
address_BLE_device_2="76F34488-4ECB-C658-B1F4-56C1AC9D741F"
address_BLE_device_3="2DAF08FB-4088-5671-EF24-0D22646F1333"

iphone_address="7C67B5F1-A5C1-E3BB-AA12-7AC52CE8B672"
list_addresses=[address_BLE_device_1,address_BLE_device_2,address_BLE_device_3,address_BLE2,address_BLE6,address_BLE10,address_BLE3]

async def scan(list_beacons):
    distance_list=[]
    for address in list_beacons:
        device = await BleakScanner.find_device_by_address(address,timeout=5)
        distance_list.append(device.rssi)
    return np.array(distance_list)

def fill_table(position, n_measurement, file=None):
    
    position_array=np.array([]).reshape(0,9)
    first=time.time()
    for i in tqdm(range(n_measurement)):
        current_measurement=asyncio.run(scan(list_beacons=list_addresses))
        while any(x>=0 for x in current_measurement):
            current_measurement=asyncio.run(scan(list_beacons=list_addresses))
        row=np.array([*current_measurement,current_x,current_y]).reshape(1,9)
        position_array=np.append(position_array,row, axis=0)
    if file == True:
        df=pd.read_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/df_fingerprint.csv')
        values_df=pd.DataFrame(position_array,index=None,columns=[list_addresses[0],list_addresses[1],list_addresses[2],list_addresses[3],list_addresses[4],list_addresses[5],list_addresses[6], 'x','y'])
        df=pd.concat([df,values_df],ignore_index=True,axis=0)
        df.to_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/df_fingerprint.csv',index=False)
    else:
        df=pd.DataFrame(position_array,index=None,columns=[list_addresses[0],list_addresses[1],list_addresses[2],list_addresses[3],list_addresses[4],list_addresses[5],list_addresses[6], 'x','y'])
        df.to_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/df_fingerprint.csv',header=True,index=False)
    last=time.time()
    print(f"Took {last-first} seconds.")

fill_table((current_x,current_y),50,file=True)
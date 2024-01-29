from locate_user_ips import *
from locate_user_ips import mv_positioning, plot_positions
from beacon import Person
from fan import *

from flask import Flask, render_template_string, request, redirect, url_for
import threading
import time

app = Flask(__name__)

# Initial value
fan_speed = 3

def get_value():
    return fan_speed

@app.route('/')
def index():
    return render_template_string('''
        <h1>Value: {{ fan_speed }}</h1>
        <form action="/increase" method="post"><button type="submit">I'm too hot</button></form>
        <form action="/decrease" method="post"><button type="submit">I'm too cold</button></form>
    ''', fan_speed=fan_speed)

@app.route('/increase', methods=['POST'])
def increase():
    global fan_speed
    if fan_speed<6:
        fan_speed += 1
        speed_change()
    return redirect(url_for('index'))

@app.route('/decrease', methods=['POST'])
def decrease():
    global fan_speed
    if fan_speed>1:
        fan_speed -= 1
        speed_change()
    return redirect(url_for('index'))
    

def run_app():
    app.run(debug=True, use_reloader=False)


def send_to_fan(person,client_socket2):
    #We define 0 as angle of fan pointing in direction of x-axis
    angle_to_send=np.rad2deg(np.arctan2((person.y-FAN_POSITION[1]),(person.x-FAN_POSITION[0])))%360 #Angle between 0 and 360
    print(f"Following person {person.id} and giving direction {angle_to_send} with speed {fan_speed}")
    message = f'(ANGLE:{360-angle_to_send})'
    client_socket2.send(message.encode())

def send_to_fan_speed():
    message = f'(SPEED:{fan_speed})'
    client_socket2.send(message.encode())

def follow(shared_data,first_position,client_socket2):
    
    while True:
        id = first_position.id
        person_to_follow=next(filter(lambda x: x.id==id,shared_data['MV']))
        first_angle=np.rad2deg(np.arctan2((first_position.y-FAN_POSITION[1]),(first_position.x-FAN_POSITION[0])))%360
        angle_to_person=np.rad2deg(np.arctan2((person_to_follow.y-FAN_POSITION[1]),(person_to_follow.x-FAN_POSITION[0])))%360
        print(f"first angle {first_angle}")
        print(f'angle to follow {angle_to_person}')
        if np.abs(first_angle-angle_to_person)>10:
            send_to_fan(person=person_to_follow,client_socket2=client_socket2)
            first_position=person_to_follow
        time.sleep(3)

def speed_change():
    print("detected speed change")
    send_to_fan_speed()

def tracking_full():
    shared_data={}
    first_position=np.array([]).reshape(0,3)
    global client_socket2
    client_socket2=init_connect_fan()
    #threading.Thread(target=locate_ble, args=(shared_data,), daemon=True).start()  
    thread_mv=threading.Thread(target=mv_positioning, args=(shared_data,False),daemon=True)
    thread_ble=threading.Thread(target=locate_ble, args=(shared_data,),daemon=True)
    
    thread_mv.start()
    thread_ble.start()

    thread_ble.join()
    print("finished locating BLE")
    threading.Thread(target=run_app, daemon=True).start()
    if 'MV' in shared_data and 'BLE' in shared_data:
        person_to_follow=match_id(shared_data)
        first_position=person_to_follow
        send_to_fan(first_position,client_socket2)
    
    thread_follow=threading.Thread(target=follow, args=(shared_data,first_position,client_socket2), daemon=True)
    
    thread_follow.start()
    plot_positions(shared_data=shared_data)
    

    
    #client_socket2.close()

def main():
    #mv_positioning(shared_data=None,plot=True)
    """current_x,current_y=9,5
    positions_list=np.array([]).reshape(0,2)
    for n,i in enumerate(range (0,5)):
        print(f"Measurement {n+1} of 100")
        position_x,position_y=locate_ble(shared_data=None,plot=False,n_measurement=10,current_position=False)
        positions_list=np.append(positions_list,[(position_x,position_y)],axis=0)
        
    
    errors=np.sqrt((current_x-positions_list[0])**2+(current_y-positions_list[1])**2)
    error=np.mean(errors,axis=0)
    np.savetxt('errors.txt',positions_list,delimiter=',', fmt='%f')
    print(f"Average error: {error}")"""
    #tracking_full()
    locate_ble(plot=True,shared_data=None,current_position=True)

if __name__=="__main__":
    main()
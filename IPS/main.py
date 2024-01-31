from locate_user_mv import *
from locate_user_mv import mv_positioning, plot_positions
from beacon import Person
import sys
#sys.path.append('/Users/guillaumelecronier/Library/Python/3.9/bin')
#sys.path.append('/Users/guillaumelecronier/anaconda3/lib/python3.11/site-packages')
from flask import Flask, render_template_string, request, redirect, url_for
import threading
import time
import sys
from fingerprint.locate_fingerprint import *

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
    if fan_speed>0:
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
        if any(person.id == first_position.id for person in shared_data['MV']):
            person_to_follow=next(filter(lambda x: x.id==first_position.id,shared_data['MV']))
        else:
            new_id=match_id(shared_data=shared_data)
            first_position.id=new_id.id
        first_angle=np.rad2deg(np.arctan2((first_position.y-FAN_POSITION[1]),(first_position.x-FAN_POSITION[0])))%360
        angle_to_person=np.rad2deg(np.arctan2((person_to_follow.y-FAN_POSITION[1]),(person_to_follow.x-FAN_POSITION[0])))%360
        print(f"first angle {first_angle}")
        print(f'angle to follow {angle_to_person}')
        if np.abs(first_angle-angle_to_person)>7:
            send_to_fan(person=person_to_follow,client_socket2=client_socket2)
            first_position=person_to_follow
        time.sleep(5)

        

def speed_change():
    print("detected speed change")
    send_to_fan_speed()

def tracking_full(ble_ips='trilateration'):
    shared_data={}
    first_position=np.array([]).reshape(0,3)
    global client_socket2
    client_socket2=init_connect_fan()
    #threading.Thread(target=locate_ble, args=(shared_data,), daemon=True).start()  
    thread_mv=threading.Thread(target=mv_positioning, args=(shared_data,False),daemon=True)
    if ble_ips=='trilateration':
        thread_ble=threading.Thread(target=locate_ble, args=(shared_data,),daemon=True)
    elif ble_ips=='fingerprinting':
        thread_ble=threading.Thread(target=locate_fingerprint, args=(shared_data,),daemon=True)
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
    #mv_positioning(shared_data=None,plot=True) #MV positioning only
    #locate_fingerprint(shared_data=None,plot_final=True) #Fingerprinting positioning only
    #locate_ble(plot=False,shared_data=None,current_position=False,final_plot=True, n_measurement=10) #BLE trilateration only
    
    #Uncomment next line for enabling all system (requiring running Pi scripts as well)
    tracking_full(ble_ips='fingerprinting') 


    #Next lines are for evaluation of positioning errors
    #evaluate_trilateration_error(file=True)
    #evaluate_fingerprint_error(file=True)
    #dict=evaluate_mv_error(current_x=10,current_y=11)
    
if __name__=="__main__":
    main()
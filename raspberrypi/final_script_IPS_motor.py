import time
import RPi.GPIO as GPIO

# Constants
DIR = 40  # Direction GPIO Pin
STEP = 38  # Step GPIO Pin
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
SPR = 200  # Steps per Revolution (360 / 1.8)
HOLD_TIME = 0.05  # Time interval for sending hold pulses (in seconds)


relay_pin_101 = 16  # GPIO23,fanspinning
pwm_pin_201 = 33  # pwm0,lights
pwm_pin_101 = 32  # pwm1,fan speed
GPIO.setwarnings(False)  # disable warnings
GPIO.setmode(GPIO.BOARD)  # set pin numbering system
GPIO.setup(pwm_pin_201, GPIO.OUT)
GPIO.setup(pwm_pin_101, GPIO.OUT)
GPIO.setup(relay_pin_101, GPIO.OUT, initial=GPIO.LOW)
pwm_101 = GPIO.PWM(pwm_pin_101, 1000)  # frequency = 5000 Hz
pwm_201 = GPIO.PWM(pwm_pin_201, 5000)  # frequency = 8000 Hz
pwm_101.start(0)  # start with 0% duty cycle
pwm_201.start(0)  # start with 0% duty cycle

import socket

# Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 12346))  # Bind to all interfaces on port 12345
server_socket.listen()

print("Waiting for connection...")
client_socket, client_address = server_socket.accept()
print(f"Connected to {client_address}")

# GPIO setup
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)

# Variable for the motor's position (in degrees)
with open('position.txt','r') as file:
    position=int(file.read())
speed=3
def set_speed(speed):

    speed=int(speed)
    # check if the device ID is vali
    if speed == 1:
        pwm_101.ChangeDutyCycle(50)
    if speed == 2:
        pwm_101.ChangeDutyCycle(60)
    if speed == 3:
        pwm_101.ChangeDutyCycle(70)
    if speed == 4:
        pwm_101.ChangeDutyCycle(75)
    if speed == 5:
        pwm_101.ChangeDutyCycle(80)
    if speed == 6:
        pwm_101.ChangeDutyCycle(90)
    elif speed == 0:
        pwm_101.ChangeDutyCycle(0)

def rotate(angle):
    global position

    # Determine direction
    

    # Calculate the number of steps and set the delay
    
    angle_difference = (angle - position + 360) % 360
    clockwise_steps = int((angle_difference / 360) * SPR)
    counter_clockwise_steps = int(SPR - clockwise_steps)

    # Determine the direction with the shortest path
    if clockwise_steps <= counter_clockwise_steps:
        dir = CW
        steps = clockwise_steps
    else:
        dir = CCW
        steps = counter_clockwise_steps
        
    GPIO.output(DIR, dir)
    
    delay = 0.0208

    # Perform the steps
    
    for _ in range(steps):
        GPIO.output(STEP, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        time.sleep(delay)

    # Update the position
    position = angle
    print('Rotation finished')
    
    with open('position.txt','w') as file:
        file.write(str(position))

def maintain_torque():
    """Sends a pulse to keep the motor energized and maintain torque."""
    GPIO.output(STEP, GPIO.HIGH)
    time.sleep(0.0001)  # Short pulse
    GPIO.output(STEP, GPIO.LOW)

# Main loop
try:
    last_move_time = time.time()
    while True:
        if time.time() - last_move_time > HOLD_TIME:
            #maintain_torque()
            print("There I am maintaining")
            last_move_time = time.time()

        message = client_socket.recv(1024).decode()
        print(message)
        if 'SPEED' in message:
            #Need to change the fan speed here
            
            start = message.find('SPEED:') + 6  # position after the colon
            end = start+1  # position of the closing parenthesis
            speed_msg = message[start:end]  # extract the substring
            speed = int(speed_msg)
            set_speed(speed)
            print(f'changing the speed to {speed}')
            
        elif 'ANGLE' in message:
            #Need to stop the fan here
            start = message.find('ANGLE:') + 6  # position after the colon
            end = message.find(')')  # position of the closing parenthesis
            angle = message[start:end]  # extract the substring
            angle = int(float(angle))  # convert to an integer
            print(f'Starting to rotate towards {angle}')
            if position!=angle:
                #set_speed(0) #Shutting down fan before turning
                #time.sleep(5) #Waiting for fan speed to be low
                rotate(angle) #Initiate rotation
                #set_speed(speed) #Set fan speed to original
            last_move_time = time.time()
            time.sleep(3)
        time.sleep(1)
        
        
except KeyboardInterrupt:
    set_speed(0)
    print("Program stopped by user")


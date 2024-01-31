import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from beacon import *
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from signal_utils import *

room_width = 8
room_length = 11
ceiling_height=1.5
FAN_POSITION=(9,4)


def plot_room_ble(position, position_weight, b1: beacon, b2: beacon, b3: beacon, x_current=0,y_current=0):
    """Plot the room and position of the marker."""
    plt.clf()  # clear the figure
    # Plot the room as a rectangle
    plt.gca().add_patch(plt.Rectangle((0, 0), room_width, room_length, fill=None))

    if x_current!=0 and y_current !=0:
        error=np.sqrt((x_current-position[0])**2+(y_current-position[1])**2)
        plt.scatter(x=room_width-x_current,y=y_current, color='green')   
        plt.text(x_current + 0.5, y_current, f'Error={str(error)}', fontsize=10)
    # Plot the current position of the marker
    plt.scatter(np.array(np.array([room_width,room_width,room_width])-[b1.x,b2.x,b3.x]),np.array([b1.y,b2.y,b3.y]),color='purple')
    position[0]=room_width-position[0]
    plt.scatter(*position, color='red')

    #position_square[1]=room_length-position_square[1]
    position_weight[0]=room_width-position_weight[0]
    #plt.scatter(*position_square, color='green')
    plt.scatter(*position_weight, color='yellow')

    for beacon in [b1, b2, b3]:
        circle = Circle((room_width-beacon.x, beacon.y), beacon.d_2D, color='blue', fill=False)
        plt.gca().add_patch(circle)

    plt.xlim(0, room_width)
    plt.ylim(0, room_length)
    plt.xticks(range(room_width + 1))
    plt.yticks(range(room_length + 1))
    plt.grid(which='both', color='black', linewidth=0.5)

    plt.draw()
    plt.pause(0.1)  # pause to allow the figure to update


"""def plot_signal(signal, signal_no_filter, n_measurement):
    distance_array=d_from_rssi(signal)
    for i in range(0,3):
        x_array=np.linspace(0,n_measurement*2,n_measurement)
        fig=make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=x_array,y=signal[:,i], name="Filtered RSSI"),row=1,col=1)
        fig.add_trace(go.Scatter(x=x_array,y=signal_no_filter[:,i], name="Unfiltered RSSI"),row=1, col=1)
        fig.add_trace(go.Scatter(x=x_array,y=distance_array[:,i], name="Distance estimation (filtered RSSI)"),row=2, col=1)
        fig.update_layout(title="Filtered and unfiltered RSSI, distance estimation in case " + str(i))
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="RSSI (dB)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Distance (m)", row=2, col=1)
        fig.show()"""



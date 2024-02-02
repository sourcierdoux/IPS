import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/df_fingerprint.csv')
coordinates=df[['x','y']].drop_duplicates()
room_width=13
room_length=17
beacon_positions=[(room_width-1.2,15.5),(room_width-1.2,1.2),(room_width-11,9.1),(room_width-4,4.5),(room_width-9.5,4.5),(room_width-6.5,9.5),(room_width-10,15)]
FAN_POSITION=(9,14)
camera_position = (2.5, 15.5)

plt.figure()
  # clear the figure

    # Plot the room as a rectangle
plt.gca().add_patch(plt.Rectangle((0, 0), room_width, room_length, fill=None))
#plt.scatter(x=room_width-coordinates['x'],y=coordinates['y'])
for x, y in beacon_positions:
    plt.scatter(x, y, color='red', marker='^')
# Remove ticks and labels for both axes
plt.scatter(x=room_width-FAN_POSITION[0],y=FAN_POSITION[1],color='brown')
plt.text(int(room_width-FAN_POSITION[0]), int(FAN_POSITION[1]), "FAN", fontsize=9, ha='right', va='bottom')
plt.scatter(x=room_width-camera_position[0],y=camera_position[1],color='black')
plt.text(int(room_width-camera_position[0]), int(camera_position[1]), "CAMERA", fontsize=9, ha='right', va='bottom')
plt.xticks(range(room_width + 1))
plt.yticks(range(room_length + 1))
plt.grid(which='both', color='black', linewidth=0.5)
plt.savefig('beacon_map.png')
plt.show()

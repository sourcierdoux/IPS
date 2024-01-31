import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/df_fingerprint.csv')
coordinates=df[['x','y']].drop_duplicates()
room_width=13
room_length=17
beacon_positions=[(room_width-1.2,15.5),(room_width-1.2,1.2),(room_width-11,9.1)]


plt.figure()
  # clear the figure

    # Plot the room as a rectangle
plt.gca().add_patch(plt.Rectangle((0, 0), room_width, room_length, fill=None))
plt.scatter(x=room_width-coordinates['x'],y=coordinates['y'])
for x, y in beacon_positions:
    plt.scatter(x, y, color='red', marker='^')
# Remove ticks and labels for both axes
plt.xticks([])
plt.yticks([])

plt.savefig('fingerprinting_map.png')
plt.show()

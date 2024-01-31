import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load the dataset
file_path = '/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/df_fingerprint.csv'  
data = pd.read_csv(file_path)

# Exploratory Data Analysis: Plotting RSSI distributions
plt.figure(figsize=(12, 6))
for column in data.columns[:-2]:  # Excluding the x and y columns
    sns.kdeplot(data[column], label=column)
plt.title('Distribution of RSSI values for each Beacon')
plt.xlabel('RSSI Value')
plt.ylabel('Density')
plt.legend()
plt.savefig('RSSI_distribution.png')
plt.show()

# Preprocessing: Normalizing the RSSI values

fan_position=(7, 14)

data['label'] = data.apply(lambda row: row['y']*13+row['x'], axis=1)

# Define the repetition number
N = 200

# Data Augmentation
augmented_data = data
"""for label in data['label'].unique():
    subset = data[data['label'] == label]
    sampled_subset = subset.sample(n=N, replace=True, random_state=42)
    augmented_data = pd.concat([augmented_data,sampled_subset],axis=0,ignore_index=True)

# Reset index in the augmented dataset
augmented_data.reset_index(drop=True, inplace=True)"""

def average_every_five_rows(df):
    # A function to process each group
    def process_group(group):
        return group.groupby(group.index // 5).mean()

    # Group by x,y and then apply the processing
    return df.groupby(['x', 'y']).apply(process_group).reset_index(drop=True)

# Apply the function to your DataFrame
averaged_df = average_every_five_rows(data)


X = averaged_df.iloc[:, :7]  # RSSI values from beacons
y = averaged_df[['x','y']]  # x and y coordinates
"""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)"""
X_scaled=X
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train=X_scaled
y_train=y
# Training the KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
random_forest=RandomForestRegressor(n_estimators=150)
random_forest.fit(X_train, y_train)

# Predicting on the test set
y_pred_knn = knn.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
#y_pred_knn_x=y_pred_knn%13
#y_pred_knn_y=y_pred_knn//13

#y_pred_rf_x=y_pred_rf%13
#y_pred_rf_y=y_pred_rf//13

#y_test_x=y_test%13
#y_test_y=y_test//13
# Evaluating the model using Mean Squared Error and Root Mean Squared Error

mse_knn = np.mean(np.sqrt((y_test['y']*0.6-y_pred_knn[:,1]*0.6)**2,(y_test['x']*0.6-y_pred_knn[:,0]*0.6)**2))


mse_rf = np.mean(np.sqrt((y_test['y']*0.6-y_pred_rf[:,1]*0.6)**2,(y_test['x']*0.6-y_pred_rf[:,0]*0.6)**2))

angle_true=np.rad2deg(np.abs(np.arctan2((fan_position[1]-y_test['y']),(fan_position[0]-y_test['x']))))
angle_pred=np.rad2deg(np.abs(np.arctan2((fan_position[1]-y_pred_rf[:,1]),(fan_position[0]-y_pred_rf[:,0]))))

print(np.mean(angle_true-angle_pred,axis=0))

# Outputting the MSE and RMSE
print("Mean Squared Error KNN:", mse_knn)


print("Mean Squared Error RF:", mse_rf)


with open('/Users/guillaumelecronier/Documents/NUS/Master thesis/IPS_BLE_MV/IPS/fingerprint/rf_model.pkl', 'wb') as file:
    pickle.dump(random_forest, file)

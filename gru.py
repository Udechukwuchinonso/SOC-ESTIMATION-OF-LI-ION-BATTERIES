import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\uchin\Downloads\SOC-and-SOH-estimation-of-lithium-ion-batteries-main\SOC-and-SOH-estimation-of-lithium-ion-batteries-main\0_deg.csv"  # Update with the correct file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to inspect the data
print(data.head())

# Select the relevant features and target
features = ['Voltage', 'Current', 'Temperature']
target = 'SOC'

# Normalize the features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Check the statistics of the scaled data
print(data.describe())
# Prepare the input and output sequences
def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # Length of the sequence
X, y = create_sequences(data[features].values, data[target].values, seq_length)

# Check the shape of the input and output sequences
print(X.shape, y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the GRU model
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(GRU(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test, label='True SoC')
plt.plot(predictions, label='Predicted SoC')
plt.legend()
plt.show()

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ipywidgets as widgets
from IPython.display import display
from joblib import dump, load

# Load the data
data = pd.read_csv("ADANIPOWER.NS.csv")

# Preprocess the data
data = data.values[-555:,:-2]
train = data[:-252,[1,4]]
test = data[-252:,[1,4]]

# Scale the data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
train_X = scaler_X.fit_transform(train[:,0].reshape(-1, 1))
train_Y = scaler_Y.fit_transform(train[:,1].reshape(-1, 1))
test_X = scaler_X.transform(test[:,0].reshape(-1, 1))
test_Y = scaler_Y.transform(test[:,1].reshape(-1, 1))

# Prepare the data for LSTM
def create_dataset(X, Y, time_step=1):
    Xs, Ys = [], []
    for i in range(len(X)-time_step):
        Xs.append(X[i:(i+time_step), 0])
        Ys.append(Y[i+time_step, 0])
    return np.array(Xs), np.array(Ys)

time_step = 10
X_train, Y_train = create_dataset(train_X, train_Y, time_step)
X_test, Y_test = create_dataset(test_X, test_Y, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, batch_size=1, epochs=1)

# Save the model
dump(model, 'lstm_model.pkl')

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler_Y.inverse_transform(train_predict)
test_predict = scaler_Y.inverse_transform(test_predict)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(test_Y[time_step:], label='True Values')
plt.plot(test_predict, label='Predicted Values')
plt.fill_between(np.arange(len(test_predict)), test_predict.flatten() - np.std(test_predict), test_predict.flatten() + np.std(test_predict), color='gray', alpha=0.2)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Stock Price Prediction with LSTM')
plt.grid()
plt.show()

# Interactive UI
def plot_predictions(timesteps):
    plt.figure(figsize=(12, 6))
    plt.plot(test_Y[time_step:], label='True Values')
    plt.plot(test_predict, label='Predicted Values')
    plt.fill_between(np.arange(len(test_predict)), test_predict.flatten() - np.std(test_predict), test_predict.flatten() + np.std(test_predict), color='gray', alpha=0.2)
    plt.xlim(0, timesteps)
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction with LSTM')
    plt.grid()
    plt.show()

timesteps_slider = widgets.IntSlider(value=50, min=10, max=len(test_predict), step=10, description='Timesteps:')
widgets.interact(plot_predictions, timesteps=timesteps_slider)

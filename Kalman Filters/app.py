import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import itertools
from tqdm import tqdm
import time
from scipy.special import logsumexp
import pickle

# Class for HMM model
class HMM_model:
    def __init__(self, n_components):
        self.n_components = n_components
        self.model = GaussianHMM(n_components=self.n_components, covariance_type='full', n_iter=1000)

    # Function to fit the model
    def fit(self, data, sequence_len):
        if sequence_len == -1:
            sequence_len = len(data)
        X = transform_data(data)
        lengths = divide_to_seq(X, sequence_len)
        self.model.fit(X, lengths)  

    # Function to predict the model
    def predict(self, train, test, latency=10):
        X = transform_data(np.vstack((train, test)))
        lengths = divide_to_seq(X, len(train))
        _, Z = self.model.decode(X, lengths)
        predictions = []
        for i in range(len(test) - latency):
            pred = np.zeros(3)
            for j in range(latency):
                pred += X[len(train) + i + j] * self.model.means_[Z[len(train) + i + j]]
            predictions.append(pred / latency)
        return np.array(predictions)[:, 0]

# Function to filter using Kalman Filter
def filtering(x, y, a, b, c, f):
    T = len(x)
    mu = np.zeros(T)
    p = np.zeros(T)
    mu[0] = x[0]
    p[0] = f
    for i in range(1, T):
        mu[i] = a + b * mu[i-1]
        p[i] = c + b**2 * p[i-1]
        K = p[i] / (p[i] + f)
        mu[i] = mu[i] + K * (y[i] - mu[i])
        p[i] = (1 - K) * p[i]
    return mu, p

# Function to transform data
def transform_data(data):
    frac_change = (data[:, 3] - data[:, 0]) / data[:, 0]
    frac_high = (data[:, 1] - data[:, 0]) / data[:, 0]
    frac_low = (data[:, 0] - data[:, 2]) / data[:, 0]
    return np.vstack((frac_change, frac_high, frac_low)).T

# Function to divide data into sequences
def divide_to_seq(data, len_):
    if len_ == 0:
        return []
    n = int(len(data) / len_)
    lengths = []
    for i in range(n):
        elt = data[i * len_: i * len_ + len_]
        lengths.append(len(elt))
    if n * len_ != len(data):
        elt = data[n * len_:]
        lengths.append(len(elt))
    return lengths

# Function to initialize parameters
def initialize_params(x, y):
    T = len(y)
    x_t = x[1:]
    x_t_1 = x[:-1]
    f_2 = ((y - x) ** 2).sum() / len(y)
    b = ((T - 1) * (x_t * x_t_1).sum() - x_t.sum() * x_t_1.sum()) / ((T - 1) * ((x_t_1 ** 2).sum()) - x_t_1.sum() ** 2)
    a = (x_t - b * x_t_1).sum() / (T - 1)
    c_2 = ((x_t - a - b * x_t_1) ** 2).sum() / (T - 1)
    return a, b, c_2, f_2

# Function to save the HMM model
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Function to load the HMM model
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Main Streamlit App
st.title("Stock Price Prediction using HMM and Kalman Filter")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Ensure all columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)  # Drop rows with non-numeric values
    
    data = data.values[-100:, :-2]
    train = data[:-50, 1:5]
    test = data[-50:, 1:5]
    train_X = train[:, 0]
    train_Y = train[:, 3]
    test_X = test[:, 0]
    test_Y = test[:, 3]

    # Parameters for HMM
    st.sidebar.header("HMM Parameters")
    n_components = st.sidebar.slider("Number of Components", min_value=2, max_value=10, value=5)
    latency = st.sidebar.slider("Latency", min_value=1, max_value=20, value=10)
    sequence_len = st.sidebar.slider("Sequence Length", min_value=-1, max_value=50, value=-1)

    # Parameters for Kalman Filter
    st.sidebar.header("Kalman Filter Parameters")
    a = st.sidebar.number_input("Parameter a", value=0.0)
    b = st.sidebar.number_input("Parameter b", value=0.0)
    c_2 = st.sidebar.number_input("Parameter c^2", value=1.0)
    f_2 = st.sidebar.number_input("Parameter f^2", value=1.0)

    if st.sidebar.button("Run Models"):
        # HMM Model
        st.header("HMM Model")
        hmm_model = HMM_model(n_components)

        # Check if a saved model exists and load it
        try:
            hmm_model = load_model('hmm_model.pkl')
            st.write("Loaded pre-trained HMM model.")
        except FileNotFoundError:
            hmm_model.fit(train, sequence_len)
            save_model(hmm_model, 'hmm_model.pkl')
            st.write("Trained and saved new HMM model.")

        predictions = hmm_model.predict(train, test, latency=latency)
        predicted_close = predictions * test[:, 0] + test[:, 0]
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(test[:50])), test[:50, 3], label='True values')
        plt.plot(np.arange(len(predicted_close[:50])), predicted_close[:50], label='Predicted values HMM')
        plt.xlabel('Timesteps')
        plt.ylabel('Close price')
        plt.legend()
        plt.grid()
        st.pyplot(plt.gcf())

        # Kalman Filter Model
        st.header("Kalman Filter Model")
        x_train = train_Y
        y_train = train_X
        x_test = test_Y
        y_test = test_X
        since_kf = time.time()
        mu_fore, p_fore = filtering(x_test, y_test, a, b, np.sqrt(c_2), np.sqrt(f_2))
        time_elapsed_kf = time.time() - since_kf
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(50), x_test[:50], label='True values')
        plt.plot(np.arange(50), mu_fore[:50], label='Predicted values KF')
        plt.fill_between(np.arange(50), mu_fore[:50] - p_fore[:50] / 2, mu_fore[:50] + p_fore[:50] / 2, color='gray', alpha=0.2)
        plt.xlabel('Timesteps')
        plt.ylabel('Close price')
        plt.legend()
        plt.grid()
        st.pyplot(plt.gcf())

        # Calculate and display metrics
        hmm_error = np.mean(np.abs((test[:, 3] - predicted_close) / test[:, 3])) * 100
        kf_error = np.mean(np.abs((x_test - mu_fore) / x_test)) * 100
        st.write(f"HMM Model Error: {hmm_error:.2f}%")
        st.write(f"Kalman Filter Model Error: {kf_error:.2f}%")
        st.write(f"Running time of Kalman Filter: {time_elapsed_kf:.2f} seconds")

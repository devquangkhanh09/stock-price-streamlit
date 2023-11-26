import time
import requests
import streamlit as st
import pandas as pd
from PIL import Image
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib.animation import FuncAnimation
import math 
from sklearn.metrics import mean_squared_error
import plotly.express as px

import sys
# setting path
sys.path.append('../services')

from services.postgresql import PostgresqlService, STOCK_INDICES

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

st.set_page_config(page_title="Prediction", page_icon=":tada:", layout="wide")

postgresql_service = PostgresqlService()

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("./style/style.css")

# ---- LOAD ASSETS ----


# ---- HEADER SECTION ----
with st.container():
    st.title("Stock price visualization & prediction")

# ---- SELECT STOCK INDEX (IMPORT DATA) ----
is_index_selected = False

with st.container():
    st.write("---")
    st.header("Stock index:")
    selected_index = st.selectbox("Please select stock index", list(STOCK_INDICES.keys()), index=None, placeholder="Select index...")
    if selected_index is not None:
        df = postgresql_service.get_stock_data_as_df(selected_index)
        df = df.filter(['Date', 'Close'])
        st.write("Load data successful " + time.strftime("%H:%M:%S"))
        is_index_selected = True


# # ---- VISUALIZE DATA ----
# with st.container():
#     st.write("---")
#     st.header("Data visualization")
#     if is_index_selected == True:
#         fig = plt.figure(figsize=(12,6))
#         plt.title('Close price History')
#         plt.plot(df['Close'])
#         plt.xlabel('Date', fontsize=18)
#         plt.ylabel('Close price $', fontsize=18)
#         st.pyplot(fig)

# ---- VISUALIZE DATA ----
with st.container():
    st.write("---")
    st.header("Data visualization")
    if is_index_selected == True:
        fig = px.line(df, x="Date", y="Close", title='Close price History')
        st.plotly_chart(fig)

        

# ---- MODEL PREDICTION ----
with st.container():
    st.write("---")
    st.header("Predict price")
    if is_index_selected == True:
        if st.button("Click here to start model"):
            placeholder = st.empty()
            placeholder.write("Processing (Data pre-processing)...")

            # ---- DATA PROCESSING ---- 
            dataset = df.values
            training_data_len = int(np.ceil(len(dataset) * .9))
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)
            train_data = scaled_data[0:int(training_data_len), :]
            x_train = []
            y_train = []

            for i in range(5, len(train_data)):
                x_train.append(train_data[i-5:i, 0])
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            # ---- BUILD MODEL ---- 
            model = Sequential()
            model.add(LSTM(256, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(LSTM(128, input_shape= (x_train.shape[1], 1)))
            model.add(Dense(8))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

            # ---- TRAIN MODEL ---- 
            placeholder.write("Processing (LSTM)...")
            model.fit(x_train, y_train, batch_size=2, epochs=2, shuffle=False)
            test_data = scaled_data[training_data_len - 5: , :]
            
            # ---- EVALUATE ---- 
            # ---- LSTM Model
            test_data = scaled_data[training_data_len - 5: , :]
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(5, len(test_data)):
                x_test.append(test_data[i-5:i, 0])
                    
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            rmse = math.sqrt(mean_squared_error(y_test, predictions))

            train = df[:training_data_len]
            valid = df[training_data_len:]
            valid['Predictions'] = predictions
            lstm_fig = plt.figure(figsize=(12,6))
            plt.title('LSTM Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price $', fontsize=18)
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

            st.write("---")
            st.subheader("Prediction by LSTM Model:")    
            st.write('Test RMSE: %.3f' % rmse)
            st.pyplot(lstm_fig)

            # ---- ARIMA Model
            placeholder.write("Processing (ARIMA)...")

            from statsmodels.tsa.arima.model import ARIMA
            X = df.values
            train, test = X[:training_data_len], X[training_data_len:]
            history = [x for x in train]
            predictions = list()

            for t in range(len(test)):
                model = ARIMA(history, order=(5,1,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            
            rmse = math.sqrt(mean_squared_error(test, predictions))
            
            df_valid = df[training_data_len:]
            df_valid['Predictions'] = predictions
            arima_fig = plt.figure(figsize=(12,6))
            plt.title('ARIMA Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price $', fontsize=18)
            plt.plot(df['Close'])
            plt.plot(df_valid[['Close', 'Predictions']])
            plt.legend(['Val', 'Predictions'], loc='lower right')

            st.write("---")
            st.subheader("Prediction by ARIMA Model:")
            st.write('Test RMSE: %.3f' % rmse)
            st.pyplot(arima_fig)

            placeholder.empty()

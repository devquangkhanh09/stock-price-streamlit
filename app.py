import time
import requests
import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
from PIL import Image
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM
from matplotlib.animation import FuncAnimation
import math 
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

st.set_page_config(page_title="TimeSeriesForecasting", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# ---- LOAD ASSETS ----


# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi, we are Group 8 from Big Data Club:wave:")
    st.title("Time Series Forecasting - Stock Prediction")
    st.write(
        "This product is used for Stock Prediction."
    )
    st.write("[See our poster >](https://drive.google.com/file/d/1rS9oSHcqHdxmL7maoLwdHriDLVE8TcyE/view?usp=sharing)")
    

# ---- ABOUT PROJECT ----
with st.container():
    st.write("---")
    st.header("About our project")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("In the current era of rapid digital transformation, the rejuvenation in the investment world has been witnessed in many markets, today's young generation thanks to technology should have access to investment earlier, as well as there have innovative and modern changes in investment thinking. Securities is one of the emerging investment fields and attracts the attention of many investors because of its huge profit potential. However, this field also requires investors to have certain knowledge and understanding, as well as implicit many risks. ")
        st.write("Therefore, stock market prediction can be of great help to investors in limiting risks, as well as helping them make the best investment decisions. Forecasting the stock market is particularly difficult by reason of the nonlinear, volatile, and complex nature of the market. Currently, stock forecasting models often fall into traditional linear models and models represented by Deep Learning. However, because data from the stock market is a time-series data with both linear and nonlinear parts, the single forecast results through forecasting models are often not so reliable.")
        st.write("Time-Series Forecasting is an important area of ​​Machine Learning because there are many prediction problems involving components of time. However, Time-Series Forecasting is often overlooked because it is the components of time that make time series problems more difficult to deal with.")
        st.write("##")
        # st.write("Will be add soon"
        # )
        st.write("[See our report >](https://drive.google.com/file/d/16_IbuXiqWthJYM--fNYwMRiOlqi7aI9m/view?usp=sharing)")
        st.write("[See our slide >](https://drive.google.com/file/d/1Fr4hF8aIY1f9V0pbFHw1MD2Ahw-Jt95R/view?usp=sharing)")
    with right_column:
        st.image('image/3.jpg')

# ---- IMPORT DATA ----
flag = False
with st.container():
    st.write("---")
    st.header("Import data:")
    uploaded_file = st.file_uploader("Please import as CSV file")
    if uploaded_file is not None:
        st.write("Load data successful " + time.strftime("%H:%M:%S"))
        flag = True
        df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date').filter(['Price'])
        df = df.iloc[::-1]


# ---- VISUALIZE DATA ----
with st.container():
    st.write("---")
    st.header("Data visualization")
    if flag == True:
        if st.button("Click here to view data"):
            st.write(df) 
        if st.button("Click here to visualize your data"):
            fig = plt.figure(figsize=(12,6))
            plt.title('Price History')
            plt.plot(df['Price'])
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price x1000 VND', fontsize=18)
            st.pyplot(fig)
        
        

# ---- MODEL PREDICTION ----
with st.container():
    st.write("---")
    st.header("Predict price")
    if flag == True:
        if st.button("Click here to start model"):
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
            from keras.layers import SimpleRNN
            model = Sequential()
            model.add(LSTM(256, return_sequences=True, input_shape= (x_train.shape[1], 1)))
            model.add(LSTM(128, input_shape= (x_train.shape[1], 1)))
            model.add(Dense(8))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

            # ---- TRAIN MODEL ---- 
            model.fit(x_train, y_train, batch_size=2, epochs=2, shuffle=False)
            test_data = scaled_data[training_data_len - 5: , :]
            
            # ---- EVALUATE ---- 
            # ---- LSTM Model
            st.write("---")
            st.subheader("Prediction by LSTM Model:")
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
            st.write('Test RMSE: %.3f' % rmse)
            train = df[:training_data_len]
            valid = df[training_data_len:]
            valid['Predictions'] = predictions
            lstm_fig = plt.figure(figsize=(12,6))
            plt.title('LSTM Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price x1000 VND', fontsize=18)
            plt.plot(train['Price'])
            plt.plot(valid[['Price', 'Predictions']])
            plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
            st.pyplot(lstm_fig)

            # ---- ARIMA Model
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
            st.write("---")
            st.subheader("Prediction by ARIMA Model:")
            rmse = math.sqrt(mean_squared_error(test, predictions))
            st.write('Test RMSE: %.3f' % rmse)
            df_valid = df[training_data_len:]
            df_valid['Predictions'] = predictions
            arima_fig = plt.figure(figsize=(12,6))
            plt.title('ARIMA Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Price x1000 VND', fontsize=18)
            plt.plot(df['Price'])
            plt.plot(df_valid[['Price', 'Predictions']])
            plt.legend(['Val', 'Predictions'], loc='lower right')
            st.pyplot(arima_fig)

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Give us your feedback!")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/phanphuocminh2002@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
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


local_css("./style/style.css")

# ---- LOAD ASSETS ----


# ---- HEADER SECTION ----
with st.container():
    st.subheader("Hi, we are Group 5 :wave:")
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
        st.image('./image/3.jpg')


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
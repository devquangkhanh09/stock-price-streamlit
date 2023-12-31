import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from datetime import date
import sys
# setting path
sys.path.append('../services')

from services.postgresql import PostgresqlService, STOCK_INDICES

postgresql_service = PostgresqlService()

# Streamlit app
st.title("Stock Data Visualization")

# Select stock index from a list of stock indices
stock_index = st.selectbox("Select stock index", list(STOCK_INDICES.keys()))
df = postgresql_service.get_stock_data_as_df(stock_index)
        


# Date range selection
start_date = st.date_input("Start Date", min(df['Date']))
end_date = st.date_input("End Date", max(df['Date']))

filter_type = st.radio(
    "How do you want to filter data?",
    ["Date range (start date & end date)", "Last week", "Last month", "Last year", "Last 5 years"])

df["Date"] = pd.to_datetime(df["Date"]).dt.date

# Filter the dataframe based on the selected date range
if filter_type != "Date range (start date & end date)":
    end_date = date.today()
    if filter_type == "Last week":
        start_date = end_date - pd.DateOffset(weeks=1)
    elif filter_type == "Last month":
        start_date = end_date - pd.DateOffset(months=1)
    elif filter_type == "Last year":
        start_date = end_date - pd.DateOffset(years=1)
    elif filter_type == "Last 5 years":
        start_date = end_date - pd.DateOffset(years=5)
    
    end_date = pd.to_datetime(end_date).date()
    start_date = pd.to_datetime(start_date).date()

filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Display the filtered dataframe
st.write("Filtered Data:")
st.write(filtered_df)

placeholder = st.empty()
with placeholder.container():
    # create three columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        label="Open",
        value=float(df['Open'][0]),
        delta=float(df['Open'][1]) - float(df['Open'][1]),
    ) 
    kpi2.metric(
        label="High",
        value=float(df['High'][0]),
        delta=float(df['High'][1]) - float(df['High'][1]),
    )
    kpi3.metric(
        label="Low",
        value=float(df['Low'][0]),
        delta=float(df['Low'][0]) - float(df['Low'][1]),
    )
    kpi4.metric(
        label="Close",
        value=float(df['Close'][0]),
        delta=float(df['Close'][0]) - float(df['Close'][1]),
    )

# Plotting using Plotly Express
fig = px.line(filtered_df, x='Date', y=['Open', 'High', 'Low', 'Close'], title='Stock Prices Over Time')
st.plotly_chart(fig)

# Additional visualization for volume
fig_volume = px.bar(filtered_df, x='Date', y='Volume', title='Volume Over Time')
st.plotly_chart(fig_volume)

# Additional visualization for candlestick chart
fig_candlestick = go.Figure(data=[go.Candlestick(x=filtered_df['Date'],
                open=filtered_df['Open'],
                high=filtered_df['High'],
                low=filtered_df['Low'],
                close=filtered_df['Close'])])
fig_candlestick.update_layout(title='Candlestick Chart'
                              , xaxis_title='Date'
                              , yaxis_title='Close')
st.plotly_chart(fig_candlestick)

# Additional visualization for bar line chart with volume (bar chart) and close price (line chart)
fig_bar_line = go.Figure()
fig_bar_line.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Volume'], name='Volume'))
fig_bar_line.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], name='Close', yaxis='y2'))
fig_bar_line.update_layout(title='Bar Line Chart'
                              , xaxis_title='Date'
                              , yaxis_title='Close'
                              , yaxis2=dict(title='Volume', overlaying='y', side='right'))
st.plotly_chart(fig_bar_line)

# Additional visualization for Correlation Matrix: 
# create a correlation matrix between Open, Close, High, Low, and Volume. 
# Calculate correlation matrix
corr_matrix = filtered_df[['Open', 'Close', 'High', 'Low', 'Volume']].corr()

# Create Heatmap
heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix, 
    x=corr_matrix.columns, 
    y=corr_matrix.columns, 
    colorscale='Viridis',
    text = np.around(corr_matrix.values, decimals=2),
    hoverongaps = False))
heatmap.update_layout(title='Correlation Matrix', xaxis_title='Columns', yaxis_title='Columns')
st.plotly_chart(heatmap)
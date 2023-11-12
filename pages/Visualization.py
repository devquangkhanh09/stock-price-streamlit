import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Sample dataset creation (replace this with your dataset)
data = {
    'Stock Index': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], # This is just a sample stock indexs
    'Date': pd.date_range(start='2023-01-01', end='2023-01-10'),
    'Price': [100, 105, 98, 102, 110, 95, 100, 108, 112, 105],
    'Open': [98, 102, 97, 100, 105, 92, 98, 104, 110, 100],
    'High': [102, 110, 100, 105, 115, 98, 104, 112, 115, 108],
    'Low': [96, 98, 95, 98, 100, 90, 94, 100, 105, 98],
    'Volume': [100000, 120000, 90000, 110000, 130000, 85000, 95000, 115000, 125000, 105000]
}
  

df = pd.DataFrame(data)


stock_indices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

df_expanded = pd.DataFrame(columns=df.columns)

for i in range(100):
    row = pd.DataFrame(columns=df.columns)
    row['Stock Index'] = [np.random.choice(stock_indices)]
    row['Date'] = [pd.to_datetime('2023-01-01') + pd.Timedelta(days=i)]
    row['Price'] = [np.random.randint(90, 115)]
    row['Open'] = [np.random.randint(85, 120)]
    row['High'] = [np.random.randint(row['Open'], 125)]
    row['Low'] = [np.random.randint(80, row['Open'])]
    row['Volume'] = [np.random.randint(90000, 130000)]
    df_expanded = pd.concat([df_expanded, row], ignore_index=True)

df = df_expanded
# Streamlit app
st.title("Stock Data Visualization")

# Select stock index from a list of stock indices
stock_index = st.selectbox("Select stock index", stock_indices)

# Date range selection
start_date = st.date_input("Start Date", min(df['Date']))
end_date = st.date_input("End Date", max(df['Date']))


df["Date"] = pd.to_datetime(df["Date"]).dt.date

# Filter the dataframe based on the selected date range
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Stock Index'] == stock_index)]

# Display the filtered dataframe
st.write("Filtered Data:")
st.write(filtered_df)

# Plotting using Plotly Express
fig = px.line(filtered_df, x='Date', y=['Price', 'Open', 'High', 'Low'], title='Stock Prices Over Time')
st.plotly_chart(fig)

# Additional visualization for volume
fig_volume = px.bar(filtered_df, x='Date', y='Volume', title='Volume Over Time')
st.plotly_chart(fig_volume)
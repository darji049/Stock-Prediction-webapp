# import all lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  
import streamlit as st
import pandas_datareader as data
import yfinance as yf
from datetime import date

st.title("Stock Market Prediction App using LSTM")

# Input field for the user to enter a stock ticker

# Date range for fetching stock data
start = st.date_input("Select Start Date:", value=pd.to_datetime("2015-01-01"))
end = st.date_input("Select End Date:", value=date.today())

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock ticker' , 'TSLA')
# Fetch the data
df = yf.download(user_input, start=start, end=end)

# discribing data 
st.subheader('Data From 2011 - 2023')
st.write(df.describe())

#Visualizations
st.subheader('Closinprice vs time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close, 'r', label = 'Closeing price')
plt.legend()
st.pyplot(fig)

st.subheader('Closinprice vs time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'b',label = 'MA100')
plt.plot(df.Close, 'r', label = 'Closeing price')
plt.legend()
st.pyplot(fig)

st.subheader('Closinprice vs time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100, 'b',label = 'MA100')
plt.plot(ma200, 'g', label = 'MA200')
plt.plot(df.Close, 'r', label = 'Closeing price')
plt.legend()
st.pyplot(fig)

# Spliting data into Training and testing (70% data is for traing and 30% data for testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Scaling down the data between 0 and 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

#load my model
model = load_model('train_stock_model.keras')

#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])
    
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_


scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b' , label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)




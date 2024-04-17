import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import streamlit as st
from keras.models import load_model
import yfinance as yf
from datetime import date,timedelta

today=date.today()
ten_years_ago=today-timedelta(days=365*10)
start = ten_years_ago
end = today

st.title('Stock Prices Prediction')

stock_symbol = st.text_input('Enter Stock Ticker','AAPL', key='stock_input')
# Define a function to convert input to uppercase on change
def to_uppercase():
    element = st.session_state['stock_input']
    element.value = element.value.upper()

# Add a hidden element to trigger the function on change
st.write('<input type="text" style="display: none;" onchange="' + to_uppercase.__name__ + '()">', unsafe_allow_html=True)

df = yf.download(tickers = stock_symbol, start = start, end=end )

# describing data
st.subheader('Data of Past 10 Years')
st.write(df.describe())

#visualizaions
st.subheader('closing price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

# 100 days moving average
st.subheader('closing price vs Time chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'green')
plt.plot(df.Close, 'blue')
st.pyplot(fig)

#200 days moving average
st.subheader('closing price vs Time chart with 100 and 200 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'green')
plt.plot(ma200 , 'red')
plt.plot(df.Close, 'blue')
st.pyplot(fig)


#splitting data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#load model
model = load_model('keras_model.h5')


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'blue', label = 'Original Price')
plt.plot(y_predicted , 'red' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

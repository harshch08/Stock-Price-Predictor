import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as data
import streamlit as st
from keras.models import load_model
import yfinance as yf
from datetime import date, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

today = date.today()
ten_years_ago = today - timedelta(days=365 * 10)
START = ten_years_ago
TODAY= today

st.title('Stock Prices Prediction')

stock_symbol = st.text_input('Enter Stock Ticker', 'AAPL', key='stock_input')
# Define a function to convert input to uppercase on change
def to_uppercase():
    element = st.session_state['stock_input']
    element.value = element.value.upper()

# Add a hidden element to trigger the function on change
st.write('<input type="text" style="display: none;" onchange="' + to_uppercase.__name__ + '()">', unsafe_allow_html=True)

df = yf.download(tickers=stock_symbol, start=START, end=TODAY)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Describing data
#st.subheader('Data of Past 10 Years')
#st.write(df.describe())

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(stock_symbol)
##data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


#200 days moving average
st.subheader('closing price vs Time chart with 100 and 200 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'green')
plt.plot(ma200 , 'red')
plt.plot(df.Close, 'blue')
st.pyplot(fig)




# Splitting data into Training and Testing
def split_data(data):
    training_size = int(len(data) * 0.70)
    training_data, testing_data = data[0:training_size], data[training_size:]
    return training_data, testing_data


# Preprocessing data
def preprocess_data(data):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


# Create sequences for prediction
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        sequences.append(sequence)
    return np.array(sequences)


# Load LSTM model
model = load_model('keras_model.h5')  # Assuming your trained model is saved here

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.subheader(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)



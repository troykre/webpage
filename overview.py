import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import os
import math
from datetime import datetime, timedelta
import seaborn as sns
sns.set()
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats
from scipy import stats
import pylab
import warnings
warnings.filterwarnings('ignore')

@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.Ticker(ticker).history(start=start_date, end=end_date).reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    return data

def app():
    st.title("Crypto Overview")
    st.write("Crypto Analysis ...")

    ticker = st.text_input("Enter a stock ticker", "BTC-USD")

    # Slider for selecting duration
    duration_days = st.slider("Select duration (days)", min_value=1, max_value=1825, value=547)  # 547 days = 18 months

    end_date = datetime.today()
    start_date = end_date - timedelta(days=duration_days)

    data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    st.line_chart(data.set_index('Date')['Close'])

    # Load and clean data for Bitcoin
    ticker_symbol = 'BTC-USD'
    BTC_data = load_data(ticker_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    BTC_data = BTC_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    BTC_data.set_index('Date', inplace=True)

    BTCHis = BTC_data

    # Create a duplicate DataFrame btc
    btc = BTC_data.copy()
    btc.index = pd.to_datetime(BTC_data.index)
    btc.sort_index(ascending=True, inplace=True)

    # Analysis options
    analysis_option = st.selectbox("Select Analysis", ["Regression Analysis", "Volatility Analysis", "Volume Analysis", "Yearly Trends", "Quarterly Trends"])

    if analysis_option == "Regression Analysis":
        st.subheader("Regression Analysis")

        # Perform linear regression analysis
        if len(btc) > 1:
            X = np.arange(len(btc)).reshape(-1, 1)
            y = btc['Close'].values.reshape(-1, 1)

            # Create and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Create a DataFrame to store the results
            results = pd.DataFrame({
                'Coefficient': [model.coef_[0][0]],
                'Intercept': [model.intercept_[0]],
                'R-squared': [model.score(X, y)]
            })

            # Display the regression results
            st.subheader("Regression Results")
            st.table(results)

            # Plot the regression line
            st.markdown("### Regression Line Plot")
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, label='Actual Prices', color='blue')
            plt.plot(X, model.predict(X), label='Regression Line', color='red')
            plt.xlabel('Days')
            plt.ylabel('Close Price')
            plt.legend()
            st.pyplot(plt)

    elif analysis_option == "Volatility Analysis":
        st.subheader("Volatility Analysis")

        # Calculate daily returns
        btc['Daily Returns'] = btc['Close'].pct_change()

        # Calculate rolling standard deviation (volatility) with a window of your choice (e.g., 30 days)
        volatility_window = 30
        btc['Volatility'] = btc['Daily Returns'].rolling(window=volatility_window).std()

        # Create a line chart to visualize volatility for the selected duration
        st.line_chart(btc['Volatility'])

    elif analysis_option == "Volume Analysis":
        st.subheader("Volume Analysis")

        # Create a line chart to visualize volume for the selected duration
        st.line_chart(btc['Volume'])

    elif analysis_option == "Yearly Trends":
        st.subheader("Yearly Trends")

        # Resample data to yearly frequency and calculate mean
        yearly_data = btc['Close'].resample('Y').mean()

        # Create a line chart to visualize yearly trends
        st.line_chart(yearly_data)

    elif analysis_option == "Quarterly Trends":
        st.subheader("Quarterly Trends")

        # Resample data to quarterly frequency and calculate mean
        quarterly_data = btc['Close'].resample('Q').mean()

        # Create a line chart to visualize quarterly trends
        st.line_chart(quarterly_data)

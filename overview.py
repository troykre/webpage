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
        st.subheader("Yearly Trends for Close Prices")
        
        # Add a dropdown to select the year
        #selected_year = st.selectbox("Select Year", range(2015, 2024))
        
        selected_year = "2023"
        
        # Filter data for the selected year
        year_data = BTC_data[f'{selected_year}-01-01':f'{selected_year}-12-31']

        # Create a line chart for the Close prices
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=year_data.index,
            y=year_data['Close'],
            mode='lines',
            name=f'Bitcoin Close Prices ({selected_year})',
            line=dict(color='blue')
        ))

        # Customize the appearance of the chart
        fig.update_layout(
            title=f'Bitcoin Close Price Trend ({selected_year})',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=True,
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            yaxis_gridcolor='lightgray',
        )
        # Show the chart
        st.plotly_chart(fig)   
        st.markdown(f"### Bitcoin Close Price Trend {selected_year}")
        st.markdown(f"This line chart displays the trend of Bitcoin's close prices for the year {selected_year}.")
        st.markdown("The blue line represents the close prices, and it shows how they fluctuated throughout the year.")           
        
        
        st.write("\n\n\n", "") 
        st.write("\n\n\n", "") 
              
        # Filter data for the selected year
        year_data = BTC_data[f'{selected_year}-01-01':f'{selected_year}-12-31']
        # Convert the index to a DateTimeIndex
        year_data.index = pd.to_datetime(year_data.index)
        # Calculate monthly average high, low, open, and close prices for the selected year
        monthly_averages = year_data.resample('M').mean()
        # Create a DataFrame for the monthly averages
        monthly_averages['Month'] = monthly_averages.index.strftime('%B %Y')

        
        # Create a bar graph for the monthly average High and Low prices with custom styling
        fig_high_low = go.Figure()
        # Add the High price as a bar
        fig_high_low.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['High'],
            name='High',
            marker_color='rgb(0, 102, 204)'  # Custom color for High price bars
        ))

        # Add the Low price as a bar
        fig_high_low.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['Low'],
            name='Low',
            marker_color='rgb(255, 0, 0)'  # Custom color for Low price bars
        ))

        # Customize the appearance of the High and Low chart
        fig_high_low.update_xaxes(type='category')  # Set x-axis type to category
        fig_high_low.update_layout(
            barmode='group',  # Set bar mode to group for side-by-side bars
            title=f'Monthly Average High and Low Prices for {selected_year}',
            xaxis_title='Month',
            yaxis_title='Price ($)',
            legend_title='Price Type',
            legend=dict(x=0.85, y=1.0),  # Position of the legend
        )

        # Show the High and Low chart with a description
        st.plotly_chart(fig_high_low)
        st.markdown("### Monthly Average High and Low Prices")
        st.markdown(f"This bar chart displays the monthly average High and Low prices for the year {selected_year}.")
        st.markdown("The blue bars represent High prices, and the red bars represent Low prices.")

        
        st.write("\n\n\n", "") 
        st.write("\n\n\n", "") 
        # Create a bar graph for the monthly average Open and Close prices with custom styling
        fig_open_close = go.Figure()

        # Add the Open price as a bar
        fig_open_close.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['Open'],
            name='Open',
            marker_color='rgb(0, 204, 0)'  # Custom color for Open price bars
        ))

        # Add the Close price as a bar
        fig_open_close.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['Close'],
            name='Close',
            marker_color='rgb(255, 153, 0)'  # Custom color for Close price bars
        ))

        # Customize the appearance of the Open and Close chart
        fig_open_close.update_xaxes(type='category')  # Set x-axis type to category
        fig_open_close.update_layout(
            barmode='group',  # Set bar mode to group for side-by-side bars
            title=f'Monthly Average Open and Close Prices for {selected_year}',
            xaxis_title='Month',
            yaxis_title='Price ($)',
            legend_title='Price Type',
            legend=dict(x=0.85, y=1.0),  # Position of the legend
        )

        # Show the Open and Close chart with a description
        st.plotly_chart(fig_open_close)
        st.markdown("### Monthly Average Open and Close Prices")
        st.markdown(f"This bar chart displays the monthly average Open and Close prices for the year {selected_year}.")
        st.markdown("The green bars represent Open prices, and the orange bars represent Close prices.")

    elif analysis_option == "Quarterly Trends":
        st.subheader("Quarterly Trends")

        # Resample data to quarterly frequency and calculate mean
        quarterly_data = btc['Close'].resample('Q').mean()

        # Create a line chart to visualize quarterly trends
        st.line_chart(quarterly_data)

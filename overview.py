import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import os
import math
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

    ticker = st.text_input("Enter a stock ticker", ticker_symbol)
    duration = st.selectbox("Select duration", ["1 month", "3 months", "6 months", "1 year", "18 months", "3 years", "5 years", "10 years", "Custom"])

    if duration == "Custom":
        start_date = st.date_input("Start date", datetime.today() - timedelta(days=365))
        end_date = st.date_input("End date", datetime.today())
    else:
        duration_mapping = {
            "1 month": 30,
            "3 months": 90,
            "6 months": 180,
            "1 year": 365,
            "18 months": 547,
            "3 years": 1095,
            "5 years": 1825,
            "10 years": 3650
        }
        end_date = datetime.today()
        start_date = end_date - timedelta(days=duration_mapping[duration])

    data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    st.line_chart(data.set_index('Date')['Close'])

# Load and clean data for Bitcoin
ticker_symbol = 'BTC-USD'
BTC_data = load_data(ticker_symbol, (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d'), datetime.today().strftime('%Y-%m-%d'))
BTC_data = BTC_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
BTC_data.set_index('Date', inplace=True)

BTCHis = BTC_data

# Create a duplicate DataFrame btc
btc = BTC_data.copy()
btc.index = pd.to_datetime(BTC_data.index)
btc.sort_index(ascending=True, inplace=True)

def app():
    st.title("Crypto Overview")
    st.write("Crypto Analysis ...")

    ticker = st.text_input("Enter a stock ticker", ticker_symbol)
    duration = st.selectbox("Select duration", ["1 month", "3 months", "6 months", "1 year", "18 months", "3 years", "5 years", "10 years", "Custom"])

    if duration == "Custom":
        start_date = st.date_input("Start date", datetime.today() - timedelta(days=365))
        end_date = st.date_input("End date", datetime.today())
    else:
        duration_mapping = {
            "1 month": 30,
            "3 months": 90,
            "6 months": 180,
            "1 year": 365,
            "18 months": 547,
            "3 years": 1095,
            "5 years": 1825,
            "10 years": 3650
        }
        end_date = datetime.today()
        start_date = end_date - timedelta(days=duration_mapping[duration])

    data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    st.line_chart(data.set_index('Date')['Close'])

    # Create a dropdown menu for different analysis options
    analysis_option = st.selectbox("Select Analysis", ["Volume Analysis", "Yearly Trends", "Quarterly Trends", "Regression Analysis",
                                                       "Volatility Analysis"])

    if analysis_option == "Volume Analysis":
        st.subheader("Volume Analysis per Year")

        selected_year_volume = st.selectbox("Select Year", range(2015, 2024))

        # Filter data for the selected year
        year_data_volume = BTC_data[f'{selected_year_volume}-01-01':f'{selected_year_volume}-12-31']

        # Create a line chart for daily trading volume
        fig_volume = px.line(year_data_volume, y='Volume',
                             labels={'Volume': 'Trading Volume'},
                             title=f'Volume Analysis for the Year {selected_year_volume}')

        # Customize the appearance of the chart
        fig_volume.update_xaxes(showgrid=True, gridcolor='gray')
        fig_volume.update_yaxes(showgrid=True, gridcolor='gray')
        fig_volume.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))

        # Show the chart for volume analysis
        st.plotly_chart(fig_volume)

        st.markdown("### Volume Analysis for the Selected Year")
        st.markdown("""
        ### Overview

        This line chart provides a visual analysis of the daily trading volume for Bitcoin in the selected year. It allows you to observe how trading volume has changed throughout the year.

        ### Insights

        - High trading volume periods may indicate increased market activity.
        - Low trading volume periods may suggest reduced market interest.
        """)

    
    elif analysis_option == "Yearly Trends":
        st.subheader("Yearly Trends for Close Prices")
        
        # Add a dropdown to select the year
        selected_year = st.selectbox("Select Year", range(2020, 2025))
        
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
        st.subheader("Quarterly Trends for Close Prices")
        st.subheader("Candlestick Chart for Quarters")

        # Candlestick Chart for Quarters
        years = range(2015, 2024)

        selected_year = str(st.selectbox("Select a Year", years))

        quarters = ["First Quarter (Jan-Mar)", "Second Quarter (Apr-Jun)", "Third Quarter (Jul-Sep)", "Fourth Quarter (Oct-Dec)"]

        selected_quarter = st.selectbox("Select a Quarter", quarters)

        # Define the start and end months for each quarter
        quarter_start_months = [1, 4, 7, 10]
        quarter_end_months = [3, 6, 9, 12]

        quarter_index = quarters.index(selected_quarter)
        start_month = quarter_start_months[quarter_index]
        end_month = quarter_end_months[quarter_index]

        # Calculate the last day of the end month
        last_day = 31 if end_month == 12 else (30 if end_month in [4, 6, 9, 11] else 28)

        # Filter the data for the selected quarter
        start_date_period = f"{selected_year}-{start_month:02}-01"
        end_date_period = f"{selected_year}-{end_month:02}-{last_day:02}"
        filtered_data = BTC_data[(BTC_data.index >= start_date_period) & (BTC_data.index <= end_date_period)]

        # Create a candlestick chart with custom colors
        fig = go.Figure(data=[go.Candlestick(
            x=filtered_data.index,
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            increasing_fillcolor='green',
            decreasing_fillcolor='red'
        )])

        # Customize the appearance of the chart
        fig.update_layout(
            title=f'Bitcoin Price Candlestick Chart ({selected_quarter} {selected_year})',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=True,
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            yaxis_gridcolor='lightgray',
            plot_bgcolor='black',  # Set background color
        )
        fig.update_xaxes(showgrid=True, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridcolor='gray')
        fig.update_traces(line=dict(width=0.5))  # Customize candlestick line width

        st.plotly_chart(fig)


        st.markdown("### Bitcoin Price Candlestick Chart for the Selected Quarter")
        st.markdown(f"""
        ### Overview

        This candlestick chart provides a visual representation of Bitcoin's price movement during the selected quarter of {selected_year}. It displays the open, high, low, and close prices for each day within the quarter.

        ### Insights

        - Candlesticks are color-coded for easy interpretation: green for price increases and red for price decreases.
        - Observe patterns such as doji, hammer, and shooting star to analyze price trends.

        Explore the chart to gain insights into Bitcoin's quarterly price fluctuations.

        """)

    elif analysis_option == "Regression Analysis":
        st.subheader("Regression Analysis")

        # Create a range of years from 2015 to 2023
        years = range(2015, 2024)

        # Select a year from the dropdown
        selected_year = str(st.selectbox("Select a Year", years))  # Convert to string

        # Convert the index values to strings
        btc['Date'] = btc.index.strftime('%Y-%m-%d')

        # Filter the data for the selected year
        filtered_data = btc[btc['Date'].str.startswith(selected_year)]

        # Create a DataFrame to store the results
        results = pd.DataFrame(columns=['Year', 'Coefficient', 'Intercept', 'R-squared'])

        # Perform linear regression analysis for the selected year if data is available
        if len(filtered_data) > 1:
            X = np.arange(len(filtered_data)).reshape(-1, 1)
            y = filtered_data['Close'].values.reshape(-1, 1)

            # Create and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Add the results to the DataFrame
            results = pd.concat([results, pd.DataFrame({
                'Year': [selected_year],
                'Coefficient': [model.coef_[0][0]],
                'Intercept': [model.intercept_[0]],
                'R-squared': [model.score(X, y)]
            })], ignore_index=True)

            # Display the regression results
            st.subheader(f"Regression Results for {selected_year}")
            st.table(results)

            # Plot the regression line
            st.markdown(f"### Regression Line Plot for {selected_year}")
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, label='Actual Prices', color='blue')
            plt.plot(X, model.predict(X), label='Regression Line', color='red')
            plt.xlabel('Days')
            plt.ylabel('Bitcoin Price ($)')
            plt.title(f'Regression Analysis for {selected_year}')
            plt.legend()
            plt.grid(True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()

            # Customize the appearance of the plot
            st.pyplot(plt)


            st.markdown(f"""
            ### Overview

            This regression analysis provides insights into the linear relationship between time (days) and Bitcoin's closing prices for the selected year, {selected_year}.

            ### Insights

            - The regression results table displays the coefficient, intercept, and R-squared value, indicating how well the regression line fits the data.
            - The scatter plot shows the actual Bitcoin prices (in blue), and the red line represents the regression line.
            - Analyze how well the linear regression model captures the price trend.

            Explore the results and plot to gain insights into Bitcoin's price behavior in {selected_year}.

            """)

            
    elif analysis_option == "Volatility Analysis":
        st.subheader("Volatility Analysis")

        # Create a range of years from 2015 to 2023
        years = range(2015, 2024)
        # Select a year from the dropdown
        selected_year = st.selectbox("Select a Year", years)

        # Filter the data for the selected year
        year_data = btc[btc.index.year == selected_year]

        # Calculate daily returns
        year_data['Daily Returns'] = year_data['Close'].pct_change()

        # Calculate rolling standard deviation (volatility) with a window of your choice (e.g., 30 days)
        volatility_window = 30
        year_data['Volatility'] = year_data['Daily Returns'].rolling(window=volatility_window).std()

        # Create a line chart to visualize volatility for the selected year
        fig_volatility = px.line(year_data, x=year_data.index, y='Volatility',
                                 title=f'Volatility Analysis for {selected_year}')
        fig_volatility.update_xaxes(title='Date')
        fig_volatility.update_yaxes(title='Volatility')
        fig_volatility.update_traces(line=dict(color='blue'))
        fig_volatility.update_layout(
            title_font=dict(size=24, color='white', family='Arial'),
            xaxis_title_font=dict(size=16, color='white', family='Arial'),
            yaxis_title_font=dict(size=16, color='white', family='Arial'),
            paper_bgcolor='black',
            plot_bgcolor='black',
        )
        st.plotly_chart(fig_volatility)

        st.markdown("---")

        st.markdown(f"""
        ### Overview

        Volatility in financial markets refers to the degree of variation or dispersion in the returns of an asset over a specific period of time. It quantifies how much the price of an asset fluctuates.

        ### Key Concepts

        - **Standard Deviation**: Volatility is often calculated using the standard deviation of an asset's returns. It represents how much the returns of an asset deviate from their average or mean.

        - **Daily Returns**: To calculate volatility, daily returns are used. Daily returns are the percentage change in the price of an asset from one day to the next.

        ### Practical Use

        - **Risk Assessment**: Volatility is a key indicator of risk. Higher volatility implies greater uncertainty and risk.

        - **Investment Strategy**: Understanding an asset's volatility helps investors choose investments that align with their risk tolerance.

        - **Options Pricing**: In options trading, volatility affects the pricing of options contracts.

        In financial analysis, volatility is crucial for assessing and managing risk, constructing diversified portfolios, and making informed investment decisions.

        """)
        
    elif analysis_option == "Other Analysis":
          st.subheader("Other Analysis")
            # Add content for other analysis options here

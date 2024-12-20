import subprocess
import sys
import streamlit as st
import overview
import metrics
import strategy

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Example usage
install('requests')
install('yfinance')
install('plotly')
install ('streamlit')
install ('numpy')
install ('pandas')
install ('matplotlib')
install ('scikit-learn')
install ('seaborn')
install ('statsmodels')
install ('scipy')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Metrics", "Strategy"])

# Navigation logic
if page == "Overview":
    overview.app()
elif page == "Metrics":
    metrics.app()
elif page == "Strategy":
    strategy.app()

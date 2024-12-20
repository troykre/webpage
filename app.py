pip install -r requirements.txt

import streamlit as st
import overview
import metrics
import strategy

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

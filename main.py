import streamlit as st
from Home import home
from aqi_data_update import get_data_update
from aqi_data_preprocessing import data_preprocessing_page
from aqi_model_finetune import fine_tune_app
from aqi_forecast import forecast_page
# from Page2 import page2

PAGES = {
    "Home": home,
    "Recent Data": get_data_update,
    "Data Processing": data_preprocessing_page, 
    "Model Fine-tuning": fine_tune_app,
    "Model Forecasting": forecast_page
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page()

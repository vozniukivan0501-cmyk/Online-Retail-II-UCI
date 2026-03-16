import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
import requests
import pandas as pd
import src.config as config

from src.Run_MDmodel import run_demand_forecast

min_date_obj = pd.to_datetime(config.min_date).date()
max_date_obj = pd.to_datetime(config.max_date).date()


date_col = st.columns(1)[0]
with date_col:
    user_date_input = st.date_input('Input current date YYYY-MM-DD',
                                    value = max_date_obj,
                                    min_value= min_date_obj,
                                    max_value=max_date_obj)
    api_date_str = f'{user_date_input} 00:00:00'

title_col, corner_col = st.columns([6, 4])

with title_col:
    st.title("Demand Forecasting")

with corner_col:
    st.metric(label="Current Engine Time", value = str(user_date_input) )

col1, col2 = st.columns(2)

with col1:
    user_n_ticks = st.number_input("Forecasting horizon (in samples)", min_value=1)

with col2:
    user_tick_size = st.number_input("Forecasting sample size (in days)", min_value=1)
    user_tick_size = user_tick_size

if st.button('Generate Forecast', type='primary'):
    with st.spinner('Generating Forecast...'):
        api_url = 'http://127.0.0.1:8080/generate_forecast'
        payload = {
            'n_ticks': user_n_ticks,
            'tick_size': user_tick_size,
            'start_date': api_date_str
        }

        df = None

        try:
            #Try hitting the API (Local Mode)
            response = requests.post(api_url, json=payload, timeout=2)
            result = response.json()

            if result.get('status') == 'complete':
                df = pd.DataFrame(result['data'])
                st.success('Forecasting complete (via FastApi)')
            else:
                st.error(f"API Error: {result.get('message')}")

        except requests.exceptions.ConnectionError:
            #Connection refused (Cloud Mode) -> Abandon API, run engine directly
            st.info("Cloud environment detected. Running engine directly")

            try:
                clean_date = api_date_str.split(' ')[0]

                df = run_demand_forecast(
                    n_ticks=user_n_ticks,
                    start_date=clean_date,
                    tick_size=user_tick_size
                )
                st.success('Forecasting complete (via Cloud Engine)')

            except Exception as e:
                st.error(f"Engine Error: {str(e)}")

        if df is not None and not df.empty:

            required_cols = ['forecast_from', 'forecast_to', 'StockCode']
            if all(col in df.columns for col in required_cols):
                df.set_index(required_cols, inplace=True)

            st.dataframe(df, use_container_width=True)



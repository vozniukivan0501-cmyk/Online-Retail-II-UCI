import pandas as pd
from src.TimeEmulator import TimeEmulator
from datetime import timedelta


from src.data_loader import PROCESSED_DATA_DIR


def run_demand_forecast(df = None,
                        n_ticks = 7,
                        start_date = "2011-09-08",
                        tick_size = 1):
    """
    Orchestrates the market demand forecasting pipeline to generate a final prediction over a specified time horizon.

        Args:
            df: DataFrame containing the complete dataset. If None, loads the default processed parquet file.
            n_ticks: The number of future time steps to forecast.
            start_date: The chronological starting point for the forecast engine (YYYY-MM-DD).
            tick_size: The duration of each forecast step in days.

        Returns:
            forecast_df: DataFrame containing the predicted sales quantities and optimal prices.
    """

    if df is None:
        df = pd.read_parquet(PROCESSED_DATA_DIR / 'online_retail.parquet')

    actual_time_delta = timedelta(days=1) * tick_size

    md_emulator = TimeEmulator(
        start_date = start_date,
        df = df
    )

    forecast_df = md_emulator.generate_forecast(n_ticks, actual_time_delta)


    return forecast_df
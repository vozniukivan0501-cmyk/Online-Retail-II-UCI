import pandas as pd
import src.config as config
from TimeEmulator import TimeEmulator
from data_loader import IPC_PATH

from data_loader import PROCESSED_DATA_DIR


def run_demand_forecast(df = pd.read_parquet(PROCESSED_DATA_DIR/ 'online_retail.parquet'), n_ticks=1):

    MD_Emulator = TimeEmulator(
        start_date = config.start_date,
        df = df
    )

    prediction = MD_Emulator.emulate_time_change(n_ticks, tick_size=config.tick_size)
    prediction.to_parquet(IPC_PATH / 'MD_prediction_buffer.parquet')

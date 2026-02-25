import config
from data_loader import MODELS_DIR, CONFIG_FILES_DIR
from src.config import start_date, tick_size
from src.ModelClasses import MarketDemandModel
import pandas as pd
import datetime

class TimeEmulator:
    def __init__(self, start_date = start_date, df = None):

        self.df = df

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        if start_date:
            self.current_datetime = pd.to_datetime(start_date)
        elif df is not None:
            self.current_datetime = df['InvoiceDate'].min()
        else:
            self.current_datetime = pd.to_datetime(start_date)

        self.history = []

        self.demand_predictor = MarketDemandModel(
            MODELS_DIR / 'MarketDemandModel.joblib',
            CONFIG_FILES_DIR / 'MarketDemandModelEncoder.joblib',
            history_df = df[df['InvoiceDate'] < self.current_datetime]
        )

        self.current_tick_data = df[df['InvoiceDate'] == self.current_datetime]

    def emulate_time_change(self, n_ticks, tick_size):
        all_predictions = []

        for _ in range(n_ticks):
            print(f'Current date:{self.current_datetime}')
            self.history.append(self.current_datetime)
            self.current_datetime += tick_size

            daily_prediction = self._on_tick()

            if not daily_prediction.empty:
                daily_df = pd.DataFrame({
                    'forecast_date': self.current_datetime,
                    'StockCode': daily_prediction.index,
                    'predicted_quantity': daily_prediction.values
                })
                all_predictions.append(daily_df)

        if all_predictions:
            full_forecast = pd.concat(all_predictions, ignore_index=True)
            full_forecast.set_index('forecast_date', inplace=True)
            return full_forecast
        else:
            return pd.DataFrame()


    def _on_tick(self):

        previous_time = self.current_datetime - pd.Timedelta(config.tick_size)

        new_sales_data = self.df[
            (self.df['InvoiceDate'] > previous_time) &
            (self.df['InvoiceDate'] <= self.current_datetime)
            ]

        MD_prediction = self.demand_predictor.predict_next_tick(new_sales_data)
        return MD_prediction
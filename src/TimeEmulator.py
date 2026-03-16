import pandas as pd
from src.data_loader import MODELS_DIR
from src.ModelClasses import MarketDemandModel


class TimeEmulator:
    """Simulates real-time forecasting operations across historical timelines"""

    def __init__(self, start_date='2011-09-08', df=None):
        """
        Initializes TimeEmulator by setting start_date as a current time and T-period in forecasting
                Args:
                    start_date: Date set as a current date for making future predictions
                    df: DataFrame containing the complete dataset. If None cold start
        """

        self.df = df

        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.start_date = pd.to_datetime(start_date)

        self.history_df = self.df[self.df['InvoiceDate'] <= self.start_date]

        self.demand_predictor = MarketDemandModel(
            MODELS_DIR / 'MarketDemandModel.joblib',
            history_df=self.history_df
        )

    def generate_forecast(self, n_ticks : int , tick_size):
        """
        Function is forming final DataFrame to show user a forecasting result

            Args:
                n_ticks: Number of ticks to make forecast
                tick_size: Sample size for single-tick forecast

            Returns:
                full_forecast: Ready to show concatenated DataFrame containing all needed forecasts
        """

        all_predictions = []

        for step in range(1, n_ticks + 1):

            target_forecast_date = self.start_date + (tick_size * step)

            demand_prediction, optimal_prices_series = self.demand_predictor.predict_future_target(
                target_date=target_forecast_date
            )

            if not demand_prediction.empty:

                qty_array = demand_prediction.values

                daily_df = pd.DataFrame({
                    'forecast_from': [(target_forecast_date - tick_size).strftime('%Y-%m-%d')] * len(qty_array),
                    'forecast_to': [target_forecast_date.strftime('%Y-%m-%d')] * len(qty_array),
                    'StockCode': demand_prediction.index,
                    'predicted_quantity': demand_prediction.clip(lower=0).values,
                    'optimal_price': optimal_prices_series.clip(lower=0.1).values,
                })
                all_predictions.append(daily_df)

        if all_predictions:
            full_forecast = pd.concat(all_predictions, ignore_index=True)
            return full_forecast

        return pd.DataFrame()
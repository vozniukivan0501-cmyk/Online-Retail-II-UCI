import joblib
import pandas as pd
import numpy as np
from src.features import MarketDemandModel_data_transformation, find_optimal_price


class MarketDemandModel:
    """2-step engine to generate market demand forecast and optimal price for all forecasting products from current to target date"""
    def __init__(self, model_path : str, history_df = None):

        """
        Initializes MarketDemandModel by loading trained ML model and recent historical data
            Args:
                model_path: path to trained model saved in .joblib format
                history_df: historical transaction DataFrame used to initialize the feature state to avoid cold start
        """

        self.model = joblib.load(model_path)

        if history_df is not None:
            self.history_buffer = history_df.copy()
        else:
            self.history_buffer = pd.DataFrame() #Cold start


    def predict_future_target(self, target_date):

        """
        Making optimal price and market demand forecast for a single target_date
                Args:
                    target_date: date set by user, the forecast end date
                Returns:
                    output: pandas series of {StockCode : predicted quantity} pairs with StockCode as an index
                    optimal_prices_series: pandas series of {StockCode : optimal_prices} pairs
        """

        last_known_state = self.history_buffer.groupby('StockCode').last().reset_index()

        #Generates placeholder records for the target forecast date, carrying forward the most recent known features
        future_dummy_df = last_known_state.copy()
        future_dummy_df['InvoiceDate'] = pd.to_datetime(target_date)

        if 'Quantity' in future_dummy_df.columns:
            future_dummy_df['Quantity'] = np.nan

        self.history_buffer = pd.concat([self.history_buffer, future_dummy_df], ignore_index=True)


        X_transformed, y_transformed = MarketDemandModel_data_transformation(
            self.history_buffer,
            is_inference=True
        )

        X_today_full = X_transformed.groupby('StockCode').last().reset_index()
        raw_stockcodes = X_today_full['StockCode'].copy()

        X_predict = X_today_full[self.model.feature_name()]

        if X_predict.empty:
            print(f'X_predict empty for {target_date}, something went wrong in transformation')
            return pd.Series(dtype='float64'), pd.Series(dtype='float64')

        #Making first forecast, using expm1 to get raw non-logarithmic values of sales quantity
        predictions = self.model.predict(X_predict)
        predictions = (np.expm1(np.clip(predictions, a_min=0, a_max=None)))
        predictions = np.round(predictions)

        output = pd.Series(predictions.flatten(), index=raw_stockcodes.values)

        target_ts = pd.to_datetime(target_date)

        prediction_map = dict(zip(raw_stockcodes.values, predictions.flatten()))

        #Updates future state placeholders with finalized model predictions
        mask = (self.history_buffer['InvoiceDate'] == target_ts) & \
               (self.history_buffer['StockCode'].isin(raw_stockcodes))
        self.history_buffer.loc[mask, 'Quantity'] = self.history_buffer.loc[mask, 'StockCode'].map(prediction_map)

        optimal_prices = {}

        #Run 2-nd step of forecasting. Optimizing revenue with searching optimal price and quantity values
        for idx in X_today_full.index:


            actual_stock_code = raw_stockcodes.iloc[idx]

            current_state_row = X_today_full.loc[[idx]].copy()

            #Defines a window where optimal price can be to avoid extreme price volatility
            current_price = current_state_row['Price'].iloc[0]
            min_test_price = current_price * 0.8
            max_test_price = current_price * 1.2

            best_price, expected_rev = find_optimal_price(
                model=self.model,
                current_state_row=current_state_row,
                min_price=min_test_price,
                max_price=max_test_price
            )

            optimal_prices[actual_stock_code] = best_price

        optimal_prices_series = pd.Series(optimal_prices, name='optimal_price')

        return output, optimal_prices_series
import joblib
import pandas as pd
import numpy as np
from data_loader import MODELS_DIR, CONFIG_FILES_DIR
from features import MarketDemandModel_data_transformation
from src.config import tick_size


class MarketDemandModel:
    def __init__(self, model_path, enc_path, history_df = None):

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(enc_path)

        if history_df is not None:
            self.history_buffer = history_df.copy()
        else:
            self.history_buffer = pd.DataFrame() #Cold start

    def predict_next_tick(self, current_tick_raw_data):
        self.history_buffer = pd.concat([self.history_buffer, current_tick_raw_data])

        X_transformed, y_transformed = MarketDemandModel_data_transformation(
            self.history_buffer,
            is_inference=True
        )

        X_today = X_transformed[y_transformed.isna().values]

        X_today =  self.encoder.transform(X_today)
        X_today = X_today.rename(columns = {'StockCode' : 'StockCode_enc'})

        X_today = X_today[self.model.feature_name()]

        if X_today.empty:
            print('X_today empty something went wrong')
            return pd.Series(dtype='float64')

        X_today = X_transformed.groupby('StockCode').last().reset_index()

        raw_stockcodes = X_today['StockCode'].copy()

        encoder = joblib.load(CONFIG_FILES_DIR / 'MarketDemandModelEncoder.joblib')
        X_today = encoder.transform(X_today)
        X_today.rename(columns={'StockCode': 'StockCode_enc'}, inplace=True)

        X_today = X_today[self.model.feature_name()]

        predictions = np.round(self.model.predict(X_today))

        predictions = np.clip(predictions, a_min=0, a_max=None)

        output = pd.Series(predictions, index=raw_stockcodes.values)

        return output
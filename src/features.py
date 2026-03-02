import holidays.countries
import pandas as pd
from pathlib import Path
import joblib
import math
import numpy as np
import holidays

import config
from src.config import tick_size
from src.data_loader import CONFIG_FILES_DIR

def get_quantile_timestamp(df, quantile):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    stamp = df['InvoiceDate'].quantile(quantile)
    return stamp


def MarketDemandModel_data_transformation(df, tick_size = config.tick_size, is_inference=False):
    if 'target_quant' in df.columns or 'ticks_since_last_sale' in df.columns:
        print('Data transformation already done')
        return df.drop(columns=['target_quant'], errors='ignore'), df.get('target_quant')
    else:
        df = df.copy()

        if df.empty:
            print("No valid sales data to transform. Bypassing.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        df = df[(df['Quantity'] > 0) & (df['Description'] != 'None') & (df['Price'] >= 0)]

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.set_index('InvoiceDate', inplace=True)
        uk_holidays = holidays.UnitedKingdom()

        df['Revenue'] = df['Price'] * df['Quantity']

        resampled_df = df.groupby('StockCode').resample(tick_size).agg({
            'Quantity': 'sum',
            'Revenue': 'sum',
            'Customer ID': 'nunique'
        }).rename(columns={'Customer ID': 'Unique_Customers'})

        resampled_df['Quantity'] = resampled_df['Quantity'].fillna(0)
        resampled_df['Revenue'] = resampled_df['Revenue'].fillna(0)

        resampled_df['Holiday_flag'] = resampled_df.index.get_level_values('InvoiceDate').map(lambda x: x in uk_holidays).astype(int)

        resampled_df['Price'] = resampled_df['Revenue'] / resampled_df['Quantity']
        resampled_df['Price'] = resampled_df.groupby(level='StockCode')['Price'].ffill().fillna(0)

        resampled_df['target_quant'] = resampled_df.groupby('StockCode')['Quantity'].shift(-1)

        product_medians = resampled_df.groupby(level='StockCode')['target_quant'].transform('median')
        resampled_df = resampled_df[product_medians > 1]

        resampled_df['lag_1t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(1).fillna(0)
        resampled_df['lag_2t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(2).fillna(0)
        resampled_df['lag_3t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(3).fillna(0)
        resampled_df['lag_4t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(4).fillna(0)

        resampled_df['EWMA_Target'] = (
            resampled_df.groupby(level='StockCode')['Quantity']
            .transform(lambda x: x.ewm(span=4).mean())
        )

        rolling_price = resampled_df.groupby(level='StockCode')['Price'].transform(lambda x: x.ewm(span=4).mean())
        resampled_df['Price_Discount_Ratio'] = resampled_df['Price'] / (rolling_price + 1e-5)

        resampled_df['Month_sin'] = np.sin(resampled_df.index.get_level_values('InvoiceDate').month * 2 * math.pi / 12)
        resampled_df['Month_cos'] = np.cos(resampled_df.index.get_level_values('InvoiceDate').month * 2 * math.pi / 12)

        if not is_inference:
            resampled_df.dropna(subset=['target_quant'], inplace=True)

        has_sale = resampled_df['Quantity'] > 0
        sale_blocks = has_sale.groupby(level='StockCode').cumsum()
        resampled_df['ticks_since_last_sale'] = resampled_df.groupby(['StockCode', sale_blocks]).cumcount()

        resampled_df.reset_index(level='StockCode', inplace=True)
        X_train = resampled_df.drop(columns='target_quant')
        X_train['StockCode'] = X_train['StockCode'].astype('category')
        y_train = np.log1p(resampled_df['target_quant'])

        X_train = X_train.sort_index()
        y_train = y_train.sort_index()

        return X_train, y_train

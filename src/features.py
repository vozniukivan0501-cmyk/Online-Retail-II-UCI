import holidays.countries
import pandas as pd
import math
import numpy as np
import holidays


def get_quantile_timestamp(df : pd.DataFrame, quantile : float):
    """Function to get upper quantile timestamp
            Args:
                df: DataFrame with 'InvoiceDate' column
                quantile: Quantile to split by
            Returns:
                stamp: Border date between upper quantile and lower [1-quantile] DataFrame parts"""
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    stamp = df['InvoiceDate'].quantile(quantile)
    return stamp


def MarketDemandModel_data_transformation(df : pd.DataFrame , tick_size : int = 7, is_inference=False):
    """Transforming raw transaction logs into time-windowed feature matrices
            Args:
                df: DataFrame to transform (for inference or for training)
                tick_size: number of days in single forecasting period
                is_inference: disables dropping rows with unknown targets when predicting future dates
            Returns:
                X : aggregated features DataFrame grouped by StockCode with InvoiceDate as an index
                y : target which is <t+1> Quantity value"""

    if 'target_quant' in df.columns or 'ticks_since_last_sale' in df.columns:
        print('Data transformation already done')
        return df.drop(columns=['target_quant'], errors='ignore'), df.get('target_quant')
    else:
        df = df.copy()

        if df.empty:
            print("No valid sales data to transform. Bypassing.")
            return pd.DataFrame(), pd.Series(dtype='float64')

        #Data-cleaning block to drop all invalid data with negative price, non-purchase data, etc
        df = df[(df['Quantity'] > 0) & (df['Description'] != 'None') & (df['Price'] >= 0)]
        df = df[df['StockCode'].str.count(r'[C]') < 1]
        df = df[df['StockCode'].str.count(r'[a-zA-Z]') <= 1]

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['InvoiceDate'] = df['InvoiceDate'].dt.normalize()
        df.set_index('InvoiceDate', inplace=True)
        uk_holidays = holidays.UnitedKingdom()

        df['Revenue'] = df['Price'] * df['Quantity']

        #Aggregates daily transaction logs into standardized time windows (ticks) to establish baseline demand metrics per product
        resampled_df = df.groupby('StockCode').resample(f'{tick_size}D').agg({
            'Quantity': 'sum',
            'Revenue': 'sum',
            'Customer ID': 'nunique'
        }).rename(columns={'Customer ID': 'Unique_Customers'})

        resampled_df['Quantity'] = resampled_df['Quantity'].fillna(0)
        resampled_df['Revenue'] = resampled_df['Revenue'].fillna(0)

        resampled_df['Holiday_flag'] = resampled_df.index.get_level_values('InvoiceDate').map(lambda x: x in uk_holidays).astype(int)

        resampled_df['Price'] = resampled_df['Revenue'] / resampled_df['Quantity']
        resampled_df['Price'] = resampled_df.groupby(level='StockCode')['Price'].ffill().fillna(0)

        #Target set as <time + 1> quantity value for current product
        resampled_df['target_quant'] = resampled_df.groupby('StockCode')['Quantity'].shift(-1)

        #Drop all the products sold rarely (sale median < 1)
        product_medians = resampled_df.groupby(level='StockCode')['target_quant'].transform('median')
        resampled_df = resampled_df[product_medians > 1]

        #Recency lag features for historical context
        resampled_df['lag_1t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(1).fillna(0)
        resampled_df['lag_2t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(2).fillna(0)
        resampled_df['lag_3t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(3).fillna(0)
        resampled_df['lag_4t'] = resampled_df.groupby(level='StockCode')['Quantity'].shift(4).fillna(0)

        #Rolling-mean trend feature with historical context for each product (weighted by recency)
        resampled_df['EWMA_Target'] = (
            resampled_df.groupby(level='StockCode')['Quantity']
            .transform(lambda x: x.ewm(span=12).mean())
        )

        #Price features block to give model context about each products trend, price changes
        resampled_df['Price_vs_Avg'] = resampled_df.groupby(level='StockCode')['Price'].transform(
            lambda x: x.rolling(8, min_periods=1).mean())
        resampled_df['Price_x_Trend'] = resampled_df['Price'] * resampled_df['EWMA_Target']
        resampled_df['Price_fall_flag'] = (resampled_df['Price'] < resampled_df.groupby(level='StockCode')['Price'].shift(1)).astype(int)
        resampled_df['Price_rise_flag'] = (resampled_df['Price'] > resampled_df.groupby(level='StockCode')['Price'].shift(1)).astype(int)
        resampled_df['Price_Shock'] = resampled_df['Price'] / resampled_df.groupby(level='StockCode')['Price'].shift(1)

        resampled_df['Month_sin'] = np.sin(resampled_df.index.get_level_values('InvoiceDate').month * 2 * math.pi / 12)
        resampled_df['Month_cos'] = np.cos(resampled_df.index.get_level_values('InvoiceDate').month * 2 * math.pi / 12)

        if not is_inference:
            resampled_df.dropna(subset=['target_quant'], inplace=True)

        has_sale = resampled_df['Quantity'] > 0
        sale_blocks = has_sale.groupby(level='StockCode').cumsum()
        resampled_df['ticks_since_last_sale'] = resampled_df.groupby(['StockCode', sale_blocks]).cumcount()

        resampled_df.reset_index(level='StockCode', inplace=True)
        X = resampled_df.drop(columns='target_quant')

        #Set StockCode as category for not to work with 5000 unique values
        X['StockCode'] = X['StockCode'].astype('category')
        #Using log1p for target to fight peaks
        y = np.log1p(resampled_df['target_quant'])

        X = X.sort_index()
        y = y.sort_index()

        return X, y



def find_optimal_price(model, current_state_row, min_price, max_price, steps=50):
    """
    Engine second-step function is maximizing revenue, gives optimal price and quantity forecast for each product
            Args:
                model: self.model in MarketDemandModel class
                current_state_row: single row about single StockCode row
                min_price: minimal historical price for current StockCode
                max_price: maximal historical price for current StockCode
                steps: number of steps to look for optimal price

            Returns:
                optimal_price: optimal price for current StockCode for current StockCode
                max_projected_revenue: revenue expected with optimal price for current StockCode
    """

    #Creating a discrete space for model to give model a possibility to make predictions with different prices to find the best of it
    test_prices = np.linspace(min_price, max_price, steps)

    simulation_df = pd.concat([current_state_row] * steps, ignore_index=True)
    simulation_df['Price'] = test_prices

    predicted_quantities = np.round(model.predict(simulation_df))
    predicted_quantities = np.clip(predicted_quantities, a_min=0, a_max=None)
    expected_revenue = test_prices * predicted_quantities

    best_index = np.argmax(expected_revenue)
    optimal_price = test_prices[best_index]
    max_projected_revenue = expected_revenue[best_index]

    return optimal_price, max_projected_revenue


def augment_price_elasticity(X : pd.DataFrame, y : pd.DataFrame, strength : float = 0.8 ):
    """Making price features more valuable in model's predictions to avoid price-blindness in find_optimal_price function
        Args:
            X : X output DataFrame from MarketDemandModel_data_transformation
            y : y output DataFrame from MarketDemandModel_data_transformation
            strength: strength of the logarithmic offset applied to y
        Returns:
            X_augmented : DataFrame with additional price features
            y_augmented: DataFrame with correlated to price changes target values """
    X_high = X.copy()
    X_low = X.copy()

    #Making a scenario where price rise/decrease to give model a simple way to understand how price changes affects on market demand
    X_high['Price'] = X['Price'] * 1.2
    X_low['Price'] = X['Price'] * 0.8

    y_high = y - np.log1p(strength)
    y_low = y + np.log1p(strength)

    #Adding price-features for every scenario to give model more context about price impact
    X_high['Price_vs_Avg'] = X_high.groupby('StockCode')['Price'].transform(
        lambda x: x.rolling(8, min_periods=1).mean())
    X_high['Price_x_Trend'] = X_high['Price'] * X_high['EWMA_Target']
    X_high['Price_fall_flag'] = (X_high['Price'] < X_high.groupby('StockCode')['Price'].shift(1)).astype(int)
    X_high['Price_rise_flag'] = (X_high['Price'] > X_high.groupby('StockCode')['Price'].shift(1)).astype(int)
    X_high['Price_Shock'] = X_high['Price'] / X_high.groupby('StockCode')['Price'].shift(1)
    X_low['Price_vs_Avg'] = X_low.groupby('StockCode')['Price'].transform(lambda x: x.rolling(8, min_periods=1).mean())
    X_low['Price_x_Trend'] = X_low['Price'] * X_low['EWMA_Target']
    X_low['Price_fall_flag'] = (X_low['Price'] < X_low.groupby('StockCode')['Price'].shift(1)).astype(int)
    X_low['Price_rise_flag'] = (X_low['Price'] > X_low.groupby('StockCode')['Price'].shift(1)).astype(int)
    X_low['Price_Shock'] = X_low['Price'] / X_low.groupby('StockCode')['Price'].shift(1)

    X_augmented = pd.concat([X_low, X, X_high], ignore_index=True)
    y_augmented = pd.concat([y_low, y, y_high], ignore_index=True)

    return X_augmented, y_augmented
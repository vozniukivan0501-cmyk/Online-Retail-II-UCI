from datetime import timedelta

min_date = '2009-12-01'
max_date = '2011-12-09'
tick_size = timedelta(days=1)

params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000,

    'learning_rate': 0.19904709713421861,
    'num_leaves': 106,
    'max_depth': 10,
    'min_child_samples': 10,
    'colsample_bytree': 0.7494696540438294,
    'cat_smooth': 14.439340080677592,}
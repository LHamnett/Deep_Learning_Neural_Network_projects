from pycaret.time_series import TSForecastingExperiment
import os
import pandas as pd

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ready_for_training_folder = os.path.join(base_path, 'data', 'ready_for_training')
train_path = os.path.join(ready_for_training_folder, 'tx1_train.csv')
train_df = pd.read_csv(train_path,parse_dates=['date'], index_col='date')
print(train_df.columns)

val_path = os.path.join(ready_for_training_folder, 'tx1_val.csv')
val_df = pd.read_csv(val_path,parse_dates=['date'], index_col='date')


input_cols_single_series = ['is_weekend',
       'is_public_holiday', 'days_to_next_holiday', 'wday_sin', 'wday_cos',
       'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos','year']

# possible cols
"""
Unnamed: 0.1', 'Unnamed: 0', 'FOODS_3_586_amount_sold',
       'FOODS_3_090_amount_sold', 'FOODS_3_555_amount_sold',
       'FOODS_3_252_amount_sold', 'FOODS_3_587_amount_sold', 'wm_yr_wk',
       'weekday', 'wday', 'month', 'year', 'event_name_1', 'event_type_1',
       'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI',
       'FOODS_3_090_sell_price', 'FOODS_3_252_sell_price',
       'FOODS_3_555_sell_price', 'FOODS_3_586_sell_price',
       'FOODS_3_587_sell_price', 'day_of_month', 'is_weekend',
       'is_public_holiday', 'days_to_next_holiday', 'wday_sin', 'wday_cos',
       'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos',
       'FOODS_3_586_amount_sold_imputed', 'FOODS_3_090_amount_sold_imputed',
       'FOODS_3_555_amount_sold_imputed', 'FOODS_3_252_amount_sold_imputed',
       'FOODS_3_587_amount_sold_imputed'
"""
#prediction_cols
cols_to_use_for_output_var = {
    'FOODS_3_586_amount_sold_imputed':input_cols_single_series,
    # 'FOODS_3_090_amount_sold_imputed',
    # 'FOODS_3_555_amount_sold_imputed',
    # 'FOODS_3_252_amount_sold_imputed',
    # 'FOODS_3_587_amount_sold_imputed'
}

train_subset_586 = train_df[input_cols_single_series+ ['FOODS_3_586_amount_sold']]

exp = TSForecastingExperiment()

# 5. Setup
exp.setup(
    data=train_subset_586,
    target='FOODS_3_586_amount_sold',
    session_id=42,
    fold=3,
    fh=14,               # forecasting horizon (e.g. 12 periods ahead)
    seasonal_period=30,   # e.g. weekly seasonality for daily data
)

# 6. Compare baseline models
print(exp.compare_models(sort='RMSE'))
# print(train_subset_586.head(10))

# print(train_df.head(10))
import pandas as pd
import os
import numpy as np
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
merged_path = os.path.join(base_path, 'data', 'merged','tx1_top_5_long_cal_info_sell_prices.csv')

merged_df = pd.read_csv(merged_path)
print(merged_df.head(5))

merged_df['date'] = pd.to_datetime(merged_df['date'])

merged_df['day_of_month'] = merged_df['date'].dt.day

#add features
def is_weekend(row):
    if row['wday'] in [1,2]:
        return 1
    else:
        return 0

merged_df['is_weekend'] = merged_df.apply(is_weekend, axis=1)

def is_public_holiday(row):
    if pd.notna(row['event_name_1']):
        return 1
    else:
        return 0

merged_df['is_public_holiday'] = merged_df.apply(is_public_holiday, axis=1)
holiday_dates = merged_df[merged_df['event_name_1'].notna()]['date'].unique()
holiday_dates = sorted(holiday_dates)
holiday_dates = pd.to_datetime(holiday_dates)

def days_to_next_holiday(current_date, holidays):
    future_holidays = holidays[holidays >= current_date]
    if len(future_holidays) == 0:
        return None  # is currently public holiday
    return int((future_holidays[0] - current_date).days)



merged_df['days_to_next_holiday'] = merged_df['date'].apply(lambda x: days_to_next_holiday(x, holiday_dates))


cols_to_convert_cyclical = ['wday','month','day_of_month']

def convert_to_cyclical(df, column_name, period):
    df[column_name + '_sin'] = np.round(np.sin(2 * np.pi * df[column_name] / period), 4)
    df[column_name + '_cos'] = np.round(np.cos(2 * np.pi * df[column_name] / period), 4)
    return df

convert_to_cyclical(merged_df, 'wday', 7)
convert_to_cyclical(merged_df, 'month', 12)
convert_to_cyclical(merged_df, 'day_of_month', 31)

# print(merged_df.head(10))

print(f'final columns: {merged_df.columns}')

engineered_dataset_location = os.path.join(base_path, 'data', 'engineered', 'tx1_top_5_long_engineered.csv')

if not os.path.exists(os.path.dirname(engineered_dataset_location)):
    os.makedirs(os.path.dirname(engineered_dataset_location))

merged_df.to_csv(engineered_dataset_location, index=True)
print(f'saved data with features engineered to \n{engineered_dataset_location}')



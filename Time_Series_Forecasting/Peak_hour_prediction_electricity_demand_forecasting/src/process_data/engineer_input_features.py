import pandas as pd
import numpy as np
import os
import yaml
import holidays

#make datetime features
def make_datetime_features(df):
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    #add cyclical encooding for day of week, hour, month
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    #add simple season
    df['season'] = ((df['month'] % 12 + 3) // 3).astype(int)  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    
    
    return df

US_HOLIDAYS_OF_INTEREST = [
    "New Year's Day",
    "Martin Luther King Jr. Day",
    "Washington's Birthday",
    "Memorial Day",
    "Independence Day",
    "Labor Day",
    "Columbus Day",
    "Veterans Day",
    "Thanksgiving Day",
    "Christmas Day"
]

HOLIDAY_CATEGORY_MAP = {
    # --- Winter family / major year-end holidays ---
    "Christmas Day": "winter_family",
    "Christmas Day (observed)": "winter_family",
    "New Year's Day": "winter_family",
    "New Year's Day (observed)": "winter_family",

    # --- Summer holidays (often hot-weather, outdoor activity, lower business demand) ---
    "Memorial Day": "summer",
    "Independence Day": "summer",
    "Labor Day": "summer",

    # --- Government / civic holidays (milder demand reduction, mostly weekday effects) ---
    "Martin Luther King Jr. Day": "government",
    "Washington's Birthday": "government",
    "Columbus Day": "government",
    "Veterans Day": "government",
    "Veterans Day (observed)": "government",

    # --- Unique autumn holiday ---
    "Thanksgiving Day": "thanksgiving",
}


def get_holiday_name(date, holidays_dict):
    holiday_name = holidays_dict.get(date, None)
    if holiday_name in US_HOLIDAYS_OF_INTEREST:
        return holiday_name

def find_time_to_next_holiday(date, holidays_dict):
    future_holidays = [d for d in holidays_dict.keys() if d >= date]
    if future_holidays:
        next_holiday = min(future_holidays)
        return (next_holiday - date).days
    else:
        return np.nan  

def find_time_since_last_holiday(date, holidays_dict):
    past_holidays = [d for d in holidays_dict.keys() if d <= date]
    if past_holidays:
        last_holiday = max(past_holidays)
        return (date - last_holiday).days
    else:
        return np.nan
     

def make_holiday_features(df):
    # Create US holidays object for the range of years in your data
    years = df['Date'].dt.year.unique()
    us_holidays = holidays.US(years=years)
    
    # Check if date is a holiday
    df['is_holiday'] = df['Date'].isin(us_holidays).astype(int)
    
    # Get holiday name (will be None if not a holiday)
    df['holiday_name'] = df['Date'].apply(lambda x: get_holiday_name(x.date(), us_holidays))
    
    #add holiday type category
    df['holiday_type'] = df['holiday_name'].map(HOLIDAY_CATEGORY_MAP)

    #add features to track time to and since last holiday
    df['time_to_next_holiday'] = df['Date'].apply(lambda x: find_time_to_next_holiday(x.date(), us_holidays))
    df['time_since_last_holiday'] = df['Date'].apply(lambda x: find_time_since_last_holiday(x.date(), us_holidays))
    
    return df

def make_summary_stats_station_temps(df):
    station_cols = [col for col in df.columns if col.startswith('temp_station_')]
    stations_temp_df = df[station_cols]
    df['max_station_temp'] = stations_temp_df.max(axis=1)
    df['min_station_temp'] = stations_temp_df.min(axis=1)
    df['mean_station_temp'] = stations_temp_df.mean(axis=1)
    df['std_station_temp'] = stations_temp_df.std(axis=1)
    df['temp_diff_stations'] = df['max_station_temp'] - df['min_station_temp']
    return df

def convert_df_one_hot(df, cat_columns):
    df_one_hot = pd.get_dummies(df, columns=cat_columns, drop_first=False)
    return df_one_hot



def make_all_x_features(df):
    
    df = make_datetime_features(df)
    df = make_holiday_features(df)
    df = make_summary_stats_station_temps(df)
    df = convert_df_one_hot(df, ['season', 'holiday_type','holiday_name'])
    return df

if __name__ == "__main__":

# Construct the absolute path to the CSV file
    cfg_path = os.path.join(os.path.dirname(__file__), '../../configs/default.yaml')

    with open(cfg_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    base_dir = cfg['paths']['top_level_folder_path']
    
    
    merged_dataset_path = os.path.join(base_dir, 'data/merged/combined_data.csv')
    merged_df = pd.read_csv(merged_dataset_path)

    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    engineered_features_df = make_all_x_features(merged_df)

    engineered_features_path = os.path.join(base_dir, 'data/features_added/engineered_features_data.csv')


    if not os.path.exists(os.path.dirname(engineered_features_path)):
        os.makedirs(os.path.dirname(engineered_features_path))

    engineered_features_df.to_csv(engineered_features_path, index=False)
    print("dataset with engineered features saved to", engineered_features_path)
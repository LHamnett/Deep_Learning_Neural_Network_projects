import pandas as pd
import yaml
import os


cfg_path = os.path.join(os.path.dirname(__file__), '../../configs/default.yaml')

with open(cfg_path, 'r') as stream:
    cfg = yaml.safe_load(stream)

base_dir = cfg['paths']['top_level_folder_path']
pivoted_weather_csv_path = os.path.join(base_dir, 'data/processed/pivoted_weather_data.csv')
historical_csv_path = os.path.join(base_dir, 'data/raw/load_hist_data.csv')

weather_df_pivot = pd.read_csv(pivoted_weather_csv_path)
weather_df_pivot['Date'] = pd.to_datetime(weather_df_pivot['Date'])

historical_df = pd.read_csv(historical_csv_path)
historical_df['Date'] = pd.to_datetime(historical_df['Date'])
historical_df['Hour'] = historical_df['Hour'].astype(int)

combined_df = pd.merge(historical_df, weather_df_pivot, on=['Date', 'Hour'], how='outer')

combined_csv_path = os.path.join(base_dir, 'data/merged/combined_data.csv')

if not os.path.exists(os.path.dirname(combined_csv_path)):
    os.makedirs(os.path.dirname(combined_csv_path))

combined_df.to_csv(combined_csv_path, index=False)
print("Combined data saved to ", combined_csv_path)
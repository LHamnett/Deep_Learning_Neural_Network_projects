import pandas as pd
import yaml
import os

# Construct the absolute path to the CSV file
cfg_path = os.path.join(os.path.dirname(__file__), '../../configs/default.yaml')

with open(cfg_path, 'r') as stream:
    cfg = yaml.safe_load(stream)

base_dir = cfg['paths']['top_level_folder_path']
weather_csv_path = os.path.join(base_dir, 'data/raw/weather_data.csv')

weather_df = pd.read_csv(weather_csv_path)

weather_df_pivot = weather_df.pivot_table(
    index=['Date', 'Hour'], 
    columns='Station ID', 
    values='Temperature', 
    aggfunc='first'  # Use 'first' in case of duplicate entries, or 'mean' for averaging
).add_prefix('temp_station_')

# Reset index to make Date and Hour regular columns
weather_df_pivot= weather_df_pivot.reset_index()

processed_dir = os.path.join(base_dir, 'data/processed')
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

weather_pivot_path = os.path.join(processed_dir, 'pivoted_weather_data.csv')

# Save the pivoted DataFrame to a new CSV file
weather_df_pivot.to_csv(weather_pivot_path, index=False)
print("Pivoted weather data saved to ", weather_pivot_path)
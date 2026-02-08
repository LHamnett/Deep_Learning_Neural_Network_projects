from engineer_target_features import make_highest_daily_load_feature, combine_y_columns

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

def make_daily_blocks(df, target_col='is_peak_hour_per_day', require_full_24h=True):
    X_blocks = []
    y_blocks = []
    peak_hour_blocks = []
    peak_hours_per_day_group = []
    dates = []
    bad_dates = []
    
    df = df.copy()
    # df['Date'] = pd.to_datetime(df['Date'])
    

    if 'is_peak_hour_per_day' not in df.columns:
        df = make_highest_daily_load_feature(df)

    

    # count hours per day
    hours_per_day = df.groupby('Date')['Hour'].nunique()

    # choose which dates to keep
    if require_full_24h:
        kept_dates = hours_per_day.index[hours_per_day == 24]

    # df = df[df['Date'].isin(kept_dates)]

    
    
    for date, group in df.groupby('Date'):
        
        # peak_hour = 
        # peak_hours_per_day_group.append(peak_hour)

        if len(group) != 24:
            continue

        dates.append(date)

        # Flatten 24h of selected features
        # Flatten all columns for the day (use specific feature cols if you prefer)
        X_day = group.to_numpy().flatten()
        y_day = group['is_peak_hour_per_day'].astype(int).to_numpy()  # shape (24,)

        X_blocks.append(X_day)
        y_blocks.append(y_day)  # keep as 1-D arrays

   

    X = np.vstack(X_blocks)                      # (n_days, 24 * n_cols_in_group)
    x_cols = [f"x_{i}" for i in range(X.shape[1])]

    # Build DataFrame: wide X + single object column for y vector
    df_out = pd.DataFrame(X, index=pd.to_datetime(dates), columns=x_cols)
    df_out['y_vector'] = list(y_blocks)          # each cell is a 24-length array (object dtype)

    # (optional) sanity prints
    print(f"Created {df_out.shape[0]} days; X width={X.shape[1]},  stored as object vectors.")

    return df_out

import numpy as np
import pandas as pd

def make_daily_blocks_simple(df, feature_cols=None, date_col='Date', hour_col='Hour', load_col='Load'):
    """
    Simplest: from hourly rows → per-day blocks.
      X: (n_days, 24 * len(feature_cols))  flattened features
      y: (n_days, 24)                      one-hot peak hour per day

    Assumes each kept day has exactly 24 hours. Ties pick the earliest hour.
    """
    df = df.copy()
    if feature_cols == None:
        feature_cols = df.columns

    X_list, y_list, date_list = [], [], []

    for dates, day in df.groupby(date_col):
        if len(day) != 24:
            continue  # skip incomplete days
        
        # flatten 24×F -> (24F,)
        X_list.append(day[feature_cols].to_numpy().reshape(-1))

        # one-hot for peak hour (earliest max because we sorted by Hour)
        peak_idx = day[load_col].to_numpy().argmax()
        y = np.zeros(24, dtype=int)
        y[peak_idx] = 1
        y_list.append(y)

        date_list.append(dates)

    if not X_list:
        return np.empty((0, 24*len(feature_cols))), np.empty((0, 24), dtype=int), pd.to_datetime([])

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    # Create DataFrames with date index
    x_cols = [f"{f}_hour_{i}" for i in range(1,25) for f in feature_cols]
    y_cols = [f"target_hour_{i}" for i in range(1,25)]

    df_x = pd.DataFrame(X, index=date_list, columns=x_cols)
    df_y = pd.DataFrame(y, index=date_list, columns=y_cols)

    print(f"Created {X.shape[0]} days; X shape={X.shape}, y shape={y.shape}")
    bad_prefixes = ['Date','Hour','Load','Year','hour','weekday','is_peak_hour']
    #remove cols that start with Date or hour
    df_x_cols_to_keep = [col for col in df_x.columns if not any(col.startswith(prefix) for prefix in bad_prefixes)]
    return df_x[df_x_cols_to_keep], df_y




if __name__ == "__main__":
    engineered_df = pd.read_csv('Amperon_take_home/data/features_added/engineered_features_data.csv')
    test_blocks_df = make_daily_blocks_simple(engineered_df)
    
import pandas as pd
import numpy as np

def check_hour_is_peak(group):
        binary_hour_is_peak = []
        
        
        for elem in group.values:
            if elem == max(group.values):
                binary_hour_is_peak.append(1)
            else:
                binary_hour_is_peak.append(0)

            #if more than one peak hour, choose one randomly
            idx_with_ones = [i for i, x in enumerate(binary_hour_is_peak) if x == 1]
            
            if sum(binary_hour_is_peak) > 1:
                idx_to_keep = np.random.choice(idx_with_ones)
                binary_hour_is_peak = [1 if i == idx_to_keep else 0 for i in range(len(binary_hour_is_peak))]
                
            
        return binary_hour_is_peak

def make_highest_daily_load_feature(df):
    df = df.copy()
    is_peak_vals = []
    
    for date,group in df.groupby('Date')['Load']:
        
        is_peak_vals.extend(check_hour_is_peak(group))
    df['is_peak_hour_per_day'] = is_peak_vals
    
    return df

def combine_y_columns(df, y_prefix='y_'):
    """
    Combine multiple one-hot target columns (e.g. y_0 ... y_23)
    into a single numpy array per row.
    """
    # Identify all columns that start with y_prefix
    y_cols = [c for c in df.columns if c.startswith(y_prefix)]
    
    # Combine values across those columns into 1D arrays per row
    df['y_vector'] = np.asarray(df[y_cols].values)
    
    # Optionally drop the individual y_ columns
    # df = df.drop(columns=y_cols)
    
    return df


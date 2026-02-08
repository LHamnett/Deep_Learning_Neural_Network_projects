import pandas as pd
import numpy as np
import os
from statsmodels.tsa.seasonal import STL

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engineered_dataset_location = os.path.join(base_path, 'data', 'engineered', 'tx1_top_5_long_engineered.csv')
engineered_df = pd.read_csv(engineered_dataset_location)
engineered_df['date'] = pd.to_datetime(engineered_df['date'])


vars_of_interest = ['FOODS_3_586_amount_sold', 'FOODS_3_090_amount_sold',
       'FOODS_3_555_amount_sold', 'FOODS_3_252_amount_sold',
       'FOODS_3_587_amount_sold']



def seasonal_trend_impute(ts, freq=30, short_gap=7):
    """
    NOT WORKING
    Decompose ts with STL, impute residuals, and recombine.
    
    Parameters
    ----------
    ts : pd.Series
        Time series with a DatetimeIndex and potential NaNs.
    freq : int
        Seasonal period 
    short_gap : int
        Max gap length (in days) to treat with linear interp on residual.

    Returns
    -------
    pd.Series
        The imputed series.
    """
    # stl decomp
    stl = STL(ts, period=freq, robust=True)
    res = stl.fit()
    trend = res.trend
    season = res.seasonal
    resid = res.resid

    # find areas with gaps
    mask = ts[ts==0]
    groups = (mask != mask.shift()).cumsum()
    gap_lengths = mask.groupby(groups).transform('sum')

    #impute residuals
    resid_filled = resid.copy()
    
    # short gaps -  linear interpolation
    short = (mask) & (gap_lengths <= short_gap)
    resid_filled[short] = np.nan
    resid_filled = resid_filled.interpolate(method='linear', limit=short_gap)

    # long gaps - leave residuals
    long = (mask) & (gap_lengths > short_gap)
    # resid_filled 
    
    #Recombine components
    filled = trend + season + resid_filled
    return filled

def linear_fill(ts):
    ts = ts.interpolate(method='polynomial', order=5)
    
    return ts.values

def mean_fill(ts):
    mean_value = ts.mean()
    ts.replace(0,np.nan,inplace=True)
    ts.fillna(mean_value)
    return ts.values

print('starting imputation for variables of interest...')
for var in vars_of_interest:
    print(f'Imputing {var}...')
    subset = engineered_df[['date', var]].copy()
    # subset['date'] = pd.to_datetime(subset['date'])
    subset.set_index('date', inplace=True)
    # subset[f'{var}_imputed'] = seasonal_trend_impute(subset[var], freq=30, short_gap=7)
    # subset[f'{var}_imputed'] = linear_fill(subset[var])
    subset[f'{var}_imputed'] = mean_fill(subset[var])

    processed_df = engineered_df.merge(subset[[f'{var}_imputed']], left_on='date', right_index=True, how='left')
    

#subset date to ignore large 0s at beginning
processed_df['date'] = pd.to_datetime(engineered_df['date'])

# print(engineered_df.head(10))

# print(engineered_df.columns)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_folder = os.path.join(base_path, 'data', 'processed')
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

processed_dataset_location = os.path.join(processed_folder, 'tx1_top_5_long_engineered_processed.csv')
processed_df.to_csv(processed_dataset_location, index=True)
print(f"Processed dataset saved to {processed_dataset_location}")


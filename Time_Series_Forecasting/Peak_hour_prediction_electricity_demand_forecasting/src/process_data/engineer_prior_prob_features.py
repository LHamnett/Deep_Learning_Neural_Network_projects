from engineer_target_features import make_highest_daily_load_feature
from typing import Dict

def make_unique_id_column(df, prior_dict, granularity='hour'):
    df = df.copy()
    
    if granularity == 'hour':
        #do nothing hour already present
        return df,'Hour'

    elif granularity == 'hour_and_weekday':
        df['weekday'] = df['Date'].dt.weekday
        df['hour_and_weekday_id'] = df['Hour'].astype(str) + '_' + df['weekday'].astype(str)
        return df,'hour_and_weekday_id'

    

def calculate_prior_dict(df, granularity:str='hour') -> Dict:
    '''
    df: pandas.DataFrame
    granularity: str, one of ['hour','hour_and_weekday']
        gives the lowest level of detail to calculate probabilities an hour will be peak in 24 hours
        eg hour and weekday, will create unique id combining hour and weekday
    
    outputs (prior_dict):
        dict mapping each unique combination to normalised probability from training data that hour is peak load
        eg if granularity is hour_and_weekday, look at all instances of 14:00 on a sunday across training data, 
        find the average value when that combination is the peak hour, repeat for all unique time combinations
        normalise across all combinations and return dict
    '''

    df = df.copy()
    df_with_peak_load = make_highest_daily_load_feature(df)

    if granularity == 'hour':
        grouped_df = df_with_peak_load.groupby('Hour')['is_peak_hour_per_day'].mean()
        grouped_df = grouped_df / grouped_df.sum() #normalise to sum to 1
        prior_dict = grouped_df.to_dict()
    
    elif granularity == 'hour_and_weekday':
        df_with_peak_load['weekday'] = df_with_peak_load['Date'].dt.weekday
        df_with_peak_load['hour_and_weekday'] = df_with_peak_load['Hour'].astype(str) + '_' + df_with_peak_load['weekday'].astype(str)
        grouped_df = df_with_peak_load.groupby(['hour_and_weekday'])['is_peak_hour_per_day'].mean()
        grouped_df = grouped_df / grouped_df.sum() #normalise to sum to 1
        prior_dict = grouped_df.to_dict()
    
    
    return prior_dict

def map_prior_probs(df, prior_dict, granularity='hour'):
    df = df.copy()
    if granularity == 'hour':
        df['prior_probs'] = df['Hour'].map(prior_dict)
    else:
        df,id_col_name = make_unique_id_column(df, prior_dict, granularity=granularity)
        df['prior_probs'] = df[id_col_name].map(prior_dict)
    return df
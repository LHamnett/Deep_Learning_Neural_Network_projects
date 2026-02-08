
import pandas as pd
import os
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from engineer_target_features import make_highest_daily_load_feature
from engineer_prior_prob_features import calculate_prior_dict, map_prior_probs, make_unique_id_column
from transform_data_into_blocks import make_daily_blocks, make_daily_blocks_simple

# scaler_to_use = StandardScaler()
scaler_to_use = MinMaxScaler()
# scaler_to_use = None

def apply_feature_transforms(df, 
                            granularity, 
                            x_features_to_scale,
                            prior_dict, 
                            fitted_scaler,
                            features_to_keep,
                            test_set=False):

    df = df.copy()
    if test_set == False:
        df = make_highest_daily_load_feature(df)
    if prior_dict != None:
        df = map_prior_probs(df, prior_dict, granularity=granularity)
    
    if x_features_to_scale != None and fitted_scaler != None:
        df[x_features_to_scale] = fitted_scaler.transform(df[x_features_to_scale])

    if features_to_keep != 'all':
        df = df[features_to_keep]
    
    if test_set == False:
        df_x,df_y = make_daily_blocks_simple(df)
        return df_x, df_y
    else:
        df_x,_ = make_daily_blocks_simple(df)
        return df_x,_
    # df = df[features_to_keep]
    
    
    # df_out = df_select
    


if __name__ == '__main__':

    
    cfg_path = os.path.join(os.path.dirname(__file__), '../../configs/default.yaml')

    with open(cfg_path, 'r') as stream:
        cfg = yaml.safe_load(stream)

    base_dir = cfg['paths']['top_level_folder_path']

    print('making train, val, test splits')

    #load dataset with engineered features
    engineered_features_path = os.path.join(base_dir, 'data/features_added/engineered_features_data.csv')
    engineered_features_df = pd.read_csv(engineered_features_path)
    engineered_features_df['Date'] = pd.to_datetime(engineered_features_df['Date'])
    engineered_features_df['datetime'] = engineered_features_df['Date'] + pd.to_timedelta(engineered_features_df['Hour'], unit='h')
    engineered_features_df['Year'] = engineered_features_df['datetime'].dt.year
    engineered_features_df.set_index('datetime', inplace=True)
    
    #choose years to use in each set
    train_years = cfg['data']['train_years']
    val_years = cfg['data']['val_years']
    test_years = cfg['data']['test_years']

    #one hot and cyclical encodings should not be scaled
    x_features_to_scale = cfg['x_features_to_scale']
    x_features_not_scale = cfg['x_features_to_not_scale']

    

    print('splitting by year...')
    
    #create splits
    train_df = engineered_features_df[engineered_features_df['Year'].isin(train_years)]
    val_df = engineered_features_df[engineered_features_df['Year'].isin(val_years)]
    test_df = engineered_features_df[engineered_features_df['Year'].isin(test_years)]

    #calculate prior probabilties that an hour will be peak load
    train_set_prior_granularity = cfg['train_set_prior_granularity']
    '''
    granularity refers to lowest level of prior probabilities that should be calculated
    eg if only hour, we look across entire training data and compare probability single hour is peak p (is_peak_hour | 14:00)
    if hour+day_of_week p(is_peak_hour | sunday, 14:00) is different to p (is_peak_hour | monday, 14:00) and so on
    '''
    
    print('calculating prior probs from train set')
    train_priors = calculate_prior_dict(train_df, granularity=train_set_prior_granularity) #calculate prior probabilities on train set only
    
    

    #apply prior probs
    # print('mapping prior probs to train and val sets')
    # train_df = make_prior_cols(train_df, train_priors, granularity=train_set_prior_granularity)
    # train_df = map_prior_probs(train_df, train_priors, granularity=train_set_prior_granularity)
    # val_df = map_prior_probs(val_df, train_priors, granularity=train_set_prior_granularity)
    
    #choose final set of input features and remove unneeded columns
    input_features_to_keep = x_features_to_scale + x_features_not_scale + ['prior_probs']
    #fit scaler
    train_df_scaled = train_df.copy()
    if scaler_to_use != None:
        print('Fitting standard scaler on train set')
        scaler_to_use.fit(train_df_scaled[x_features_to_scale])
        # scaler_to_use.fit(train_df_scaled)
    

    # features_to_keep = input_features_to_keep + ['is_peak_hour_per_day']
    
    #apply preprocessing
    print('converting train data into daily vectors')
    train_x_df, train_y_df = apply_feature_transforms(train_df,
                                            granularity=train_set_prior_granularity,
                                            x_features_to_scale=x_features_to_scale,
                                            prior_dict=train_priors,
                                            fitted_scaler=scaler_to_use,
                                            features_to_keep='all'
                                            )

    # train_x_df, train_y_df = apply_feature_transforms(train_df,
    #                                         granularity=train_set_prior_granularity,
    #                                         x_features_to_scale=x_features_to_scale,
    #                                         prior_dict=train_priors,
    #                                         fitted_scaler=scaler_to_use,
    #                                         features_to_keep='all'
    #                                         )

    print('converting val data into daily vectors')
    val_x_df, val_y_df  = apply_feature_transforms(val_df,
                                            granularity=train_set_prior_granularity,
                                                x_features_to_scale=x_features_to_scale,
                                                prior_dict=train_priors,
                                                fitted_scaler=scaler_to_use,
                                                features_to_keep='all'
                                                )

    # val_x_df, val_y_df  = apply_feature_transforms(val_df,
    #                                         granularity=train_set_prior_granularity,
    #                                             x_features_to_scale=None,
    #                                             prior_dict=train_priors,
    #                                             fitted_scaler=scal,
    #                                             features_to_keep='all'
    #                                             )
    
    
    test_x_df,_ = apply_feature_transforms(test_df,
                                            granularity=train_set_prior_granularity,
                                                x_features_to_scale=x_features_to_scale,
                                                prior_dict=train_priors,
                                                fitted_scaler=scaler_to_use,
                                                features_to_keep='all',
                                                test_set=True
                                                )

    
    print('finished making datasets')
    splits_folder = os.path.join(base_dir, 'data/splits')

    if not os.path.exists(splits_folder):
        os.makedirs(splits_folder)
    
    train_x_output_path = os.path.join(splits_folder, 'train_x.csv')
    train_y_output_path = os.path.join(splits_folder, 'train_y.csv')
    val_x_output_path = os.path.join(splits_folder, 'val_x.csv')
    val_y_output_path = os.path.join(splits_folder, 'val_y.csv')
    
    test_x_output_path = os.path.join(splits_folder, 'test_x.csv')

    train_x_df.to_csv(train_x_output_path, index=True)
    train_y_df.to_csv(train_y_output_path, index=True)
    val_x_df.to_csv(val_x_output_path, index=True)
    val_y_df.to_csv(val_y_output_path, index=True)

    test_x_df.to_csv(test_x_output_path, index=True)
    print('train, val, test splits saved to ', os.path.dirname(splits_folder))
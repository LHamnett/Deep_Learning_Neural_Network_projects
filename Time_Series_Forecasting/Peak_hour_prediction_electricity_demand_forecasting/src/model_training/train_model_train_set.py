from sklearn.ensemble import RandomForestRegressor
import yaml
import os
import pandas as pd

def make_preds_val_set(trained_model, x_val):
    preds = trained_model.predict(x_val)
    return preds

def process_datetime_cols(df):
    df['datetime'] = pd.to_datetime(df['Date']) + pd.to_timedelta(df['Hour'], unit='h')
    df.set_index('datetime', inplace=True)
    df.drop(columns=['Date','Hour'], inplace=True)
    return df

def create_preds_df(y_val, preds):
    preds_df = y_val.copy()
    preds_df['preds'] = preds
    preds_df.index = pd.to_datetime(y_val.index)  # Ensure datetime index
    return preds_df

#load datasets
train_data_path = 'Amperon_take_home/data/splits/train_data.csv'
train_data_df = pd.read_csv(train_data_path,index_col=0)

val_data_path = 'Amperon_take_home/data/splits/val_data.csv'
val_data_df = pd.read_csv(val_data_path,index_col=0)

cfg_path = "Amperon_take_home/configs/default.yaml"

with open(cfg_path, 'r') as stream:
    cfg = yaml.safe_load(stream)

all_input_features = cfg['x_features_to_scale'] + cfg['x_features_to_not_scale'] + cfg['train_set_calculated_features']
# train_data_df_processed = process_datetime_cols(train_data_df)
x_train = train_data_df[all_input_features]
y_train = train_data_df[['Load']]

# val_data_df_processed = process_datetime_cols(val_data_df)
x_val = val_data_df[all_input_features]
y_val_actual = val_data_df[['Load']]

print(y_val_actual.head(10))

#train regression model
print('training model on train set with shape', x_train.shape, 'and', y_train.shape)
trained_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
trained_model.fit(x_train, y_train)
print('model trained')

print('making predictions on val set with shape', x_val.shape)
#generate preds
y_val_preds = make_preds_val_set(trained_model, x_val)
val_pred_df = create_preds_df(y_val_actual, y_val_preds)

#save model and preds
# model_path = 'Amperon_take_home/models/rf_model.pkl'
# if not os.path.exists(os.path.dirname(model_path)):
#     os.makedirs(os.path.dirname(model_path))
exp_name = 'random_forest'

pred_path = f'Amperon_take_home/data/predictions/{exp_name}_val_set_preds.csv'
if not os.path.exists(os.path.dirname(pred_path)):
    os.makedirs(os.path.dirname(pred_path))

val_pred_df.to_csv(pred_path, index=True)
print('predictions saved')






from pycaret.time_series import TSForecastingExperiment as TSExp
import pandas as pd
from sklearn.metrics import root_mean_squared_error as rmse
import os
import matplotlib.pyplot as plt
import seaborn as sns


base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ready_for_training_folder = os.path.join(base_path, 'data', 'ready_for_training')
trained_model_folder = os.path.join(base_path, 'models', 'trained_models')
train_path = os.path.join(ready_for_training_folder, 'tx1_train.csv')
train_df = pd.read_csv(train_path,parse_dates=['date'], index_col='date')

val_path = os.path.join(ready_for_training_folder, 'tx1_val.csv')
val_df = pd.read_csv(val_path,parse_dates=['date'], index_col='date')

input_cols_single_series = ['is_weekend',
       'is_public_holiday', 'days_to_next_holiday', 'wday_sin', 'wday_cos',
       'month_sin', 'month_cos', 'day_of_month_sin', 'day_of_month_cos','year']

train_subset_586 = train_df[input_cols_single_series+ ['FOODS_3_586_amount_sold']]
train_subset_090 = train_df[input_cols_single_series+ ['FOODS_3_090_amount_sold']]
train_subset_555 = train_df[input_cols_single_series+ ['FOODS_3_555_amount_sold']]
train_subset_252 = train_df[input_cols_single_series+ ['FOODS_3_252_amount_sold']]
train_subset_587 = train_df[input_cols_single_series+ ['FOODS_3_587_amount_sold']]

val_subset_586 = val_df[input_cols_single_series+ ['FOODS_3_586_amount_sold']]
val_subset_090 = val_df[input_cols_single_series+ ['FOODS_3_090_amount_sold']]
val_subset_555 = val_df[input_cols_single_series+ ['FOODS_3_555_amount_sold']]
val_subset_252 = val_df[input_cols_single_series+ ['FOODS_3_252_amount_sold']]
val_subset_587 = val_df[input_cols_single_series+ ['FOODS_3_587_amount_sold']]


predictions_folder = os.path.join(base_path, 'reports','val_preds_tx1')
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)

def load_model_make_and_save_predicts(df, output_col):
    model_path = os.path.join(trained_model_folder, f'{output_col}_model')
    exp = TSExp()
    exp.setup(
        data=df,
        target=output_col,
        fold_strategy='sliding',
        fold=2,
        fh=14,
        session_id=123,
        numeric_imputation_target='linear',
    )
    model = exp.load_model(model_path)
    actual_output = val_df[output_col]
    # val_df.drop(columns=[output_col], inplace=True)
    predictions_output = exp.predict_model(estimator=model)

    val_actual = df.iloc[-14:][output_col]

    preds_df = predictions_output.copy()
    preds_df['actual'] = val_actual.values

    rmse_out = round(rmse(preds_df['y_pred'], preds_df['actual']),2)
    
    output_path = os.path.join(predictions_folder, f'{output_col}_preds_rmse_{rmse_out}.csv')
    preds_df.to_csv(output_path,index=True)
    return rmse_out, preds_df

print("Loading models and making predictions...")
rmse_586,preds_df_586 = load_model_make_and_save_predicts(train_subset_586, 'FOODS_3_586_amount_sold')
rmse_090,preds_df_090 = load_model_make_and_save_predicts(train_subset_090, 'FOODS_3_090_amount_sold')
rmse_555,preds_df_555 = load_model_make_and_save_predicts(train_subset_555, 'FOODS_3_555_amount_sold')
rmse_252,preds_df_252 = load_model_make_and_save_predicts(train_subset_252, 'FOODS_3_252_amount_sold')
rmse_587,preds_df_587 = load_model_make_and_save_predicts(train_subset_587, 'FOODS_3_587_amount_sold')
print('all predictions made against val data')

# print(preds_df_586.head(20))

predictions_folder = os.path.join(base_path, 'reports', 'val_preds_tx1')

def make_save_viz(preds_df, rmse,output_col):
    df = preds_df.copy()
    df['predicted'] = pd.to_numeric(df['y_pred'], errors='coerce')
    df['actual']    = pd.to_numeric(df['actual'], errors='coerce')
    
    
    idx = df.index
    if isinstance(idx, pd.PeriodIndex):
        dates = idx.to_timestamp()
    else:
        dates = pd.to_datetime(idx)
    

    #plotting
    plt.figure(figsize=(14, 6))
    plt.plot(dates, df['actual'],    label='Actual',    linewidth=2)
    plt.plot(dates, df['predicted'], label='Predicted', linestyle='--', linewidth=2,color='orange')
    plt.title(f'{output_col} — Actual vs Predicted\nRMSE: {rmse:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Amount Sold')
    plt.xticks(dates, rotation=45)  # Add x labels for every day
    plt.legend()
    plt.tight_layout()
    
    #save
    viz_path = os.path.join(predictions_folder, f'{output_col}_preds_viz.png')
    plt.savefig(viz_path)
    plt.show()
    plt.close()
    
    return
    

make_save_viz(preds_df_586, rmse_586, 'FOODS_3_586_amount_sold')
make_save_viz(preds_df_090, rmse_090, 'FOODS_3_090_amount_sold')
make_save_viz(preds_df_555, rmse_555, 'FOODS_3_555_amount_sold')
make_save_viz(preds_df_252, rmse_252, 'FOODS_3_252_amount_sold')
make_save_viz(preds_df_587, rmse_587, 'FOODS_3_587_amount_sold')

print('all visualizations saved')

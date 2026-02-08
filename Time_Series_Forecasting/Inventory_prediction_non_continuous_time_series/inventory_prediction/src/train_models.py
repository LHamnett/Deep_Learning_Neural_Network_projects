from pycaret.time_series import TSForecastingExperiment as TSExp
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
from sktime.forecasting.base import ForecastingHorizon

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ready_for_training_folder = os.path.join(base_path, 'data', 'ready_for_training')
train_path = os.path.join(ready_for_training_folder, 'tx1_train.csv')

train_df = pd.read_csv(train_path, parse_dates=['date'], index_col='date')
print(f"Loaded train set: {len(train_df)} rows")

# define features and targets
input_cols = [
    'is_weekend','is_public_holiday','days_to_next_holiday',
    'wday_sin','wday_cos','month_sin','month_cos',
    'day_of_month_sin','day_of_month_cos','year'
]

targets = [
    'FOODS_3_586_amount_sold',
    'FOODS_3_090_amount_sold',
    'FOODS_3_555_amount_sold',
    'FOODS_3_252_amount_sold',
    'FOODS_3_587_amount_sold',
]

models_dir = os.path.join(base_path, 'models', 'trained_models')
if not os.path.exists(models_dir):
       os.makedirs(models_dir)
       print(f"Created directory for models: {models_dir}")



def train_model_pycaret(train_df, output_col):
    # select only the exogenous + target columns
    df_in = train_df[input_cols + [output_col]].copy()
    
    exp = TSExp()
    exp.setup(
        data=df_in,
        target=output_col,
        fold_strategy='sliding',
        fold=2,
        fh=14,
        session_id=123,
        numeric_imputation_target='linear',
        enforce_exogenous=False,  # model is univariate + simple features
    )
    print(f"[Train] fitting model for {output_col}...")
    model = exp.create_model('rf_cds_dt')
    
    model_path = os.path.join(models_dir, f"{output_col}_model")
    exp.save_model(model, model_path, model_only=False)
    print(f"[Train] saved model: {model_path}")



train_model_pycaret(train_df = train_df, output_col='FOODS_3_586_amount_sold')
train_model_pycaret(train_df = train_df, output_col='FOODS_3_090_amount_sold')
train_model_pycaret(train_df = train_df, output_col='FOODS_3_555_amount_sold')
train_model_pycaret(train_df = train_df, output_col='FOODS_3_252_amount_sold')
train_model_pycaret(train_df = train_df, output_col='FOODS_3_587_amount_sold')

print('all models trained and saved')


#NOT WORKING
# def train_model_sktime(train_df, input_cols, output_col):
#     # Split into y and exogenous X
#     y_train = train_df[output_col]
#     X_train = train_df[input_cols]

#     # Build the pipeline
#     forecaster = TransformedTargetForecaster(steps=[
#         ("deseason", Deseasonalizer(model="additive", sp=30)),
#         ("detrend",  Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
#         ("forecast", make_reduction(
#             estimator=RandomForestRegressor(n_estimators=100, random_state=42),
#             strategy="recursive",
#             window_length=14
#         ))
#     ])

#     # Fit with exogenous
#     forecaster.fit(y=y_train, X=X_train)

#     # Save model
#     model_path = os.path.join(trained_model_folder, f'{output_col}_sktime_model.joblib')
#     forecaster.save(model_path)
#     print(f'sktime model {output_col} trained and saved at {model_path}')
#     return forecaster

# def predict_model_sktime(forecaster, full_df, input_cols, output_col, fh):
#     """
#     full_df must contain both the history and at least `fh` future rows
#     with exogenous columns present in `input_cols`.
#     """
#     # Extract the exogenous data for the forecast period
#     X_full = full_df[input_cols]
#     # Build forecasting horizon relative to last training point
#     fh_rel = ForecastingHorizon(range(1, fh + 1), is_relative=True)

#     # Predict, supplying X for those future steps
#     y_pred = forecaster.predict(fh=fh_rel, X=X_full)

#     # If you have actuals in full_df, extract them:
#     y_true = full_df[output_col].loc[y_pred.index]

#     return pd.DataFrame({"predicted": y_pred, "actual": y_true})

# forecaster = train_model_sktime(train_df, input_cols, output_col)
# results_df = predict_model_sktime(forecaster, full_df, input_cols, output_col, fh=14)
# print(results_df)
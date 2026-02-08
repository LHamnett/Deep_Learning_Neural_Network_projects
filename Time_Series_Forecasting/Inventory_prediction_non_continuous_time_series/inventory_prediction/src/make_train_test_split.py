import os
import pandas as pd

# Configuration
TRAIN_VAL_PERCENT = 0.9
HOLDOUT_DAYS = 14

# Paths
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_csv = os.path.join(base_path, 'data', 'processed', 'tx1_top_5_long_engineered_processed.csv')
output_dir = os.path.join(base_path, 'data', 'ready_for_training')

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
processed_df = pd.read_csv(processed_csv, parse_dates=['date'], infer_datetime_format=True)

# Split into train+val and test
n_total      = len(processed_df)
n_train_val  = int(n_total * TRAIN_VAL_PERCENT)
train_val_df = processed_df.iloc[:n_train_val].reset_index(drop=True)
test_df      = processed_df.iloc[n_train_val:].reset_index(drop=True)

# Further split train_val into train and validation
train_df = train_val_df.iloc[:-HOLDOUT_DAYS].reset_index(drop=True)
val_df   = train_val_df.iloc[-HOLDOUT_DAYS:].reset_index(drop=True)

# Report sizes
print(f"Train+Val: {len(train_val_df)} rows")
print(f"Train:   {len(train_df)} rows")
print(f"Val:     {len(val_df)} rows")
print(f"Test:     {len(test_df)} rows")

# Save splits
train_val_df.to_csv(os.path.join(output_dir, 'tx1_train_val.csv'), index=False)
train_df    .to_csv(os.path.join(output_dir, 'tx1_train.csv'),     index=False)
val_df      .to_csv(os.path.join(output_dir, 'tx1_val.csv'),       index=False)
test_df     .to_csv(os.path.join(output_dir, 'tx1_test.csv'),      index=False)

print(f"Saved train/val/test splits under {output_dir}")

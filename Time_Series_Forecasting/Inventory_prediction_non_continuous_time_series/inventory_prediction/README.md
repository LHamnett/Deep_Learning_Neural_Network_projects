# Inventory forecasting assignment

## Overview
This project is designed for inventory forecasting using the M5 wallmart stock dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data


## Project Structure
```
inventory_prediction/
├── data/  
│   ├── raw/  
│   │   └── m5-forecasting-accuracy/ – Contains the original M5 CSV files (calendar, price, and sales) as downloaded.  
│   ├── merged/  
│   │   └──  – Holds the joined sales, calendar, and price table for the five top items at store TX_1.  
│   ├── engineered/  
│   │   └──  – Adds derived date features (holiday flags, cyclical   encodings, days-to-next-holiday).  
│   ├── processed/  
│   │   └── – Applies imputation to fill zeros/missing sales in the engineered data.  
│   └── ready_for_training/  
│       └── – Splits the processed data into train, validation, and test CSVs.  
├── models/  
│   └── trained_models/ – Stores the five serialized PyCaret time-series model artifacts.  
├── notebooks/  
│   └── eda.ipynb – Jupyter notebook for exploratory data analysis and seasonality diagnostics.  
├── report_info/  
│   ├── best_models/ – Logs of model comparison results and rankings.  
│   ├── figures/ – Diagrams and all generated plots (raw series, seasonality checks, validation charts).  
│   └── val_preds_tx1/ – CSVs and visuals of 14-day validation forecasts with RMSE annotations.  
├── scripts/  
│   └── run_all.sh – Shell wrapper that executes the full pipeline end-to-end.  
├── src/  
│   ├── retrieve_data.py – Downloads/unzips the raw competition files.  
│   ├── data_loader_merger.py – Loads raw tables and merges sales, calendar, and price.  
│   ├── engineer_features.py – Implements date-based feature engineering.  
│   ├── preprocess.py – Runs imputation routines on intermittent series.  
│   ├── make_train_test_split.py – Creates train/validation/test splits.  
│   ├── train_models.py – Fits and saves the PyCaret forecasting models.  
│   ├── compare_models.py / .ipynb – Benchmarks alternative sktime/PyCaret forecasters.  
│   ├── evaluate_trained_models_vs_val_data.py – Loads models, predicts on hold-out, computes RMSE, and saves outputs.  
│   └── simulate_restocking_policy.py – Simulates an (s, S) inventory policy with cost metrics.  
├── tests/ – Contains unit and integration tests for each pipeline stage.  
├── README.md – High-level project overview and setup instructions.  

└── requirements.txt – Lists all Python dependencies needed to reproduce the analysis.  

```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd inventory_prediction
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download the raw data from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data . Extract into the data/raw folder

## Usage
- To run the entire pipeline, execute the following script:
  ```
  bash scripts/run_all.sh
  ```
- For (re)training the model, use:
  ```
  python scripts/train_model.py
  ```
- To simulate inventory management, run:
  ```
  python scripts/simulate_inventory.py
  ```

## License
Copyright (c) 2025 Leon Hamnett. All rights reserved.

Permission is hereby granted for personal, educational, or non-commercial use of this software, documentation, and related materials (the “Software”) provided that this notice is retained in all copies.

Commercial use of the Software—including distribution, modification, or incorporation into commercial products or services—is strictly prohibited without the express prior written permission of the copyright holder. Requests for commercial licensing should be directed to L.s.hamnett@gmail.com.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY ARISING FROM THE USE OR DISTRIBUTION OF THE SOFTWARE.

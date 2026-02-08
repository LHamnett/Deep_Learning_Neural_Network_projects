#!/bin/bash

# Run the entire end-to-end pipeline, assumes data has been downloaded and extracted into data/raw


#load and preprocess the data
python src/data_loader_merger.py
python src/engineer_features.py
python src/preprocess_data.py
python src/make_train_test_split.py

# train the model
python src/train_models.py

# eval the model against val sets
python src/evaluate_trained_models_vs_val_data.py

#simulate the inventory management
python scripts/simulate_inventory.py


echo "All steps completed successfully."
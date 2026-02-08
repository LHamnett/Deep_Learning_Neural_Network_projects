# main.py

import os
from src.data.preprocessing import preprocess_data
from src.models.train import train_model
from src.utils.helpers import load_config

def main():
    # Load configuration
    config = load_config('config.yaml')

    # Load and preprocess data
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')
    data = preprocess_data(raw_data_path, processed_data_path)

    # Train the model
    model = train_model(data)

    # Save the trained model
    model.save('model.pkl')

if __name__ == "__main__":
    main()
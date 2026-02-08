# src/models/train.py

import numpy as np
import pandas as pd



def load_data(data_path):
    """Load dataset from the specified path."""
    merged_df_train_save_path = os.path.join(config['top_level_folder'],'data/ready_for_training/train_gt_embed_preds.pkl')

def preprocess_data(data):
    """Preprocess the data (placeholder for actual preprocessing steps)."""
    # Implement preprocessing steps here
    return data

def train_model(X_train, y_train, model):
    """Train the machine learning model."""
    model.fit(X_train, y_train)
    return model

def validate_model(model, X_val, y_val):
    """Validate the trained model."""
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy

def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

def main(data_path, model, output_model_path):
    """Main training routine."""
    data = load_data(data_path)
    processed_data = preprocess_data(data)
    
    X = processed_data.drop('target', axis=1)  # Replace 'target' with your target column name
    y = processed_data['target']  # Replace 'target' with your target column name
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trained_model = train_model(X_train, y_train, model)
    accuracy = validate_model(trained_model, X_val, y_val)
    
    print(f'Model validation accuracy: {accuracy}')
    save_model(trained_model, output_model_path)

    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

    with open(config_path,'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
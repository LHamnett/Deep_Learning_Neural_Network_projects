def log_message(message):
    # Function to log messages
    print(f"[LOG] {message}")

def calculate_metrics(y_true, y_pred):
    # Function to calculate and return metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {"accuracy": accuracy, "f1_score": f1}

def load_config(config_file):
    # Function to load configuration from a file
    import json
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import yaml

os.makedirs('data', exist_ok=True)

params = {}
if os.path.exists('params.yaml'):
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f).get('preprocessing', {})

test_size = params.get('test_size', 0.2)
val_size = params.get('val_size', 0.25)
random_state = params.get('random_state', 42)
input_file = params.get('input_file', 'data/creditcard.csv')
scaler_output = params.get('scaler_output', 'data/scaler.pkl')
train_output = params.get('train_output', 'data/processed_train.csv')
val_output = params.get('val_output', 'data/processed_val.csv')
test_output = params.get('test_output', 'data/processed_test.csv')

def load_data(file_path):
    """Load the dataset"""
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset"""
    print("Preprocessing data")
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    # Create and fit scaler
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    X['Time'] = scaler.fit_transform(X[['Time']])
    
    # Save the scaler for future use
    with open(scaler_output, 'wb') as f:
        pickle.dump(scaler, f)
    
    return X, y

def split_data(X, y, test_size, val_size, random_state):
    """Split data into train, validation, and test sets"""
    print("Splitting data into train, validation, and test sets")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, 
        random_state=random_state, stratify=y_train_val
    )
    
    print(f"Training set size: {X_train.shape[0]}, Positive cases: {sum(y_train)}")
    print(f"Validation set size: {X_val.shape[0]}, Positive cases: {sum(y_val)}")
    print(f"Test set size: {X_test.shape[0]}, Positive cases: {sum(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save the processed datasets"""
    print("Saving processed datasets")
    
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(train_output, index=False)
    val_data.to_csv(val_output, index=False)
    test_data.to_csv(test_output, index=False)
    
    print(f"Saved processed datasets to {train_output}, {val_output}, {test_output}")

if __name__ == "__main__":
    df = load_data(input_file)
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size, val_size, random_state
    )
    save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    print("Preprocessing complete!")

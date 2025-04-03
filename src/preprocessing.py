import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def load_data(data_path):
    return pd.read_csv(data_path)

def transform_binary_columns(data, binary_cols=None):
    if binary_cols is None:
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    data_copy = data.copy()
    for col in binary_cols:
        data_copy[col] = data_copy[col].map({'yes': 1, 'no': 0, 'Male': 1, 'Female': 0})
    return data_copy

def transform_ordinal_columns(data, ordinal_mappings=None):
    if ordinal_mappings is None:
        ordinal_mappings = {
            'CAEC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'CALC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        }
    data_copy = data.copy()
    for col, mapping in ordinal_mappings.items():
        data_copy[col] = data_copy[col].map(mapping)
    return data_copy

def apply_one_hot_encoding(data, columns=None):
    if columns is None:
        columns = ['MTRANS']
    return pd.get_dummies(data, columns=columns)

def encode_target_variable(data, target_col='NObeyesdad', target_mapping=None):
    if target_mapping is None:
        target_mapping = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 6
        }
    data_copy = data.copy()
    data_copy[target_col] = data_copy[target_col].map(target_mapping)
    return data_copy

def handle_missing_values(data):
    return data.fillna(0)

def preprocess_dataset(data):
    data = transform_binary_columns(data)
    data = transform_ordinal_columns(data)
    data = apply_one_hot_encoding(data)
    data = encode_target_variable(data)
    data = handle_missing_values(data)
    return data

def split_data(data, target_col='NObeyesdad', test_size=0.3, val_size=0.5, random_state=42):
    X = data.drop(columns=[target_col])
    y = LabelEncoder().fit_transform(data[target_col])
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalize_data(X_train, X_val=None, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_val_scaled, X_test_scaled

if __name__ == "__main__":
    data = load_data("BMI.csv")
    processed_data = preprocess_dataset(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)
    X_train_scaled, X_val_scaled, X_test_scaled = normalize_data(X_train, X_val, X_test)
    joblib.dump((X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test), "processed_data.pkl")
    print("Data preprocessing complete. Saved as processed_data.pkl")

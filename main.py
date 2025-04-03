from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, log_loss, classification_report
)
from functools import lru_cache

# Initialize FastAPI app and load environment variables
app = FastAPI()
load_dotenv()

# CORS middleware for handling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
MODEL_PATH = "models/rf_model.pkl"

# BMI Classification mapping
BMI_CLASSES = {
    0: 'Insufficient_Weight',   # Less than 18.5
    1: 'Normal_Weight',        # 18.5 to 24.9
    2: 'Overweight_Level_I',   # 25 to 27.49
    3: 'Overweight_Level_II',  # 27.5 to 29.9
    4: 'Obesity_Type_I',       # 30.0 to 34.9
    5: 'Obesity_Type_II',      # 35.0 to 39.9
    6: 'Obesity_Type_III'      # Higher than 40
}

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    print("Error: Model file not found. Ensure `rf_model.pkl` is in the `models/` directory.")
    raise HTTPException(status_code=500, detail="Model file not found.")

# Create a connection function to use with MongoDB
@lru_cache(maxsize=1)
def get_mongo_client():
    """Create and return a MongoDB client (only when needed)"""
    MONGO_URI = os.getenv("MONGO_URI")
    return MongoClient(MONGO_URI)

def get_collection():
    """Get the MongoDB collection safely after worker forking"""
    client = get_mongo_client()
    db = client["BMI-Data"]
    return db["patients"]

# Remove the global MongoDB connection
# client = MongoClient(MONGO_URI)  # <-- This was causing the fork safety issue
# db = client["BMI-Data"]
# collection = db["patients"]

transport_mapping = {
    "Automobile": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Walking": 4
}

def preprocess_data(df):
    """Convert categorical variables into numerical format."""
    categorical_cols = ["CAEC", "CALC", "MTRANS"]
    
    # One-hot encode 'CAEC' and 'CALC' (if needed)
    df = pd.get_dummies(df, columns=["CAEC", "CALC"], drop_first=True)
    
    # Label encode 'MTRANS'
    if "MTRANS" in df.columns:
        df["MTRANS"] = df["MTRANS"].map(transport_mapping).fillna(0).astype(int)

    return df

# Define request model for prediction
class PredictionRequest(BaseModel):
    id: int
    Gender: int  # 1 for Male, 0 for Female
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: int  # 1 for yes, 0 for no
    FAVC: int  # Frequent consumption of high caloric food, 1 for yes, 0 for no
    FCVC: float  # Frequency of consumption of vegetables
    NCP: float  # Number of main meals
    CAEC: int  # Consumption of food between meals (0-3)
    SMOKE: int  # 1 for yes, 0 for no
    CH2O: float  # Consumption of water daily
    SCC: int  # Calories consumption monitoring, 1 for yes, 0 for no
    FAF: float  # Physical activity frequency
    TUE: float  # Time using technology devices
    CALC: int  # Consumption of alcohol (0-3)
    MTRANS: str  # Transportation used - raw categorical value

# Predict endpoint
@app.post("/predict")
def predict(data: PredictionRequest):
    model = load_model()
    
    # Convert the input data to a DataFrame
    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Extract MTRANS columns
    mtrans_cols = [col for col in input_df.columns if col.startswith("MTRANS_")]
    
    # Create a single MTRANS column from the one-hot encoded inputs
    if mtrans_cols:
        # Find which MTRANS option is set to 1
        for col in mtrans_cols:
            if input_df[col].iloc[0] == 1:
                transport_type = col.replace("MTRANS_", "")
                break
        else:
            transport_type = "Automobile"  # Default if none selected
        
        # Create the MTRANS column
        input_df["MTRANS"] = transport_type
        
        # Drop the MTRANS_* columns
        input_df = input_df.drop(columns=mtrans_cols)
    
    # Apply the same preprocessing as during training
    # First make sure CAEC and CALC are encoded as they were during training
    input_df = pd.get_dummies(input_df, columns=["CAEC", "CALC"], prefix_sep="_", dtype=float)
    
    # Handle MTRANS with one-hot encoding
    input_df = pd.get_dummies(input_df, columns=["MTRANS"], prefix_sep="_", dtype=float)
    
    # Ensure all expected columns from training are present (fill missing with 0)
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Ensure only columns used during training are included
    input_df = input_df[expected_columns]
    
    # Make prediction
    prediction_numeric = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(max(probabilities) * 100)
    
    return {
        "bmi_class": BMI_CLASSES[prediction_numeric],
        "class_number": int(prediction_numeric),
        "confidence": confidence
    }

# Retrain model endpoint with memory optimization
@app.post("/retrain")
def retrain():
    try:
        # Get collection reference safely
        collection = get_collection()
        
        # Fetch data in batches to reduce memory usage
        batch_size = 1000
        data_batches = []
        cursor = collection.find({}, {"_id": 0})
        
        # Process in manageable chunks
        current_batch = []
        for i, doc in enumerate(cursor):
            current_batch.append(doc)
            if (i + 1) % batch_size == 0:
                data_batches.append(pd.DataFrame(current_batch))
                current_batch = []
        
        # Add the last batch if it exists
        if current_batch:
            data_batches.append(pd.DataFrame(current_batch))
        
        # Check if we have any data
        if not data_batches:
            return {"message": "No new data available for retraining."}
            
        # Combine batches
        df = pd.concat(data_batches, ignore_index=True)
        
        if "NObeyesdad" not in df.columns:
            raise HTTPException(status_code=500, detail="'NObeyesdad' target column missing from dataset.")
        
        # Preprocess data
        df = preprocess_data(df)

        X = df.drop(columns=["NObeyesdad"], errors='ignore')
        y = df["NObeyesdad"].astype(int)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42

        # Train new model
        new_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            verbose=0
        )
        new_model.fit(X_train, y_train)

        # Evaluate new model
        new_preds = new_model.predict(X_test)
        new_probs = new_model.predict_proba(X_test)

        new_model_metrics = {
            "accuracy": accuracy_score(y_test, new_preds),
            "precision": precision_score(y_test, new_preds, average="weighted"),
            "recall": recall_score(y_test, new_preds, average="weighted"),
            "f1_score": f1_score(y_test, new_preds, average="weighted"),
            "log_loss": log_loss(y_test, new_probs),
            "classification_report": classification_report(y_test, new_preds, output_dict=True)
        }

        # Save new model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(new_model, MODEL_PATH)

        return {
            "message": "Model retrained successfully.",
            "old_model_metrics": old_model_metrics,
            "new_model_metrics": new_model_metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feature importance endpoint
@app.get("/feature_importance")
def get_feature_importance():
    try:
        model = load_model()
        importance = model.feature_importances_
        feature_names = model.feature_names_in_
        
        # Create sorted importance dictionary
        importance_dict = {name: float(importance[i]) for i, name in enumerate(feature_names)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {"feature_importance": sorted_importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload CSV data to MongoDB endpoint
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        # Check if required columns are present
        required_columns = ["Gender", "Age", "Height", "Weight", "NObeyesdad"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Convert binary columns to 0/1
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'Male': 1, 'Female': 0})
        
        # Convert ordinal columns
        ordinal_mappings = {
            'CAEC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'CALC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        }
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Convert target variable if needed
        target_mapping = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 6
        }
        if "NObeyesdad" in df.columns and df["NObeyesdad"].dtype == 'object':
            df["NObeyesdad"] = df["NObeyesdad"].map(target_mapping)
        
        # Get MongoDB collection safely    
        collection = get_collection()
        
        # Insert data into MongoDB
        collection.insert_many(df.to_dict(orient="records"))
        return {"message": f"Data uploaded successfully. Added {len(df)} records."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

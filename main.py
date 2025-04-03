from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Constants
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

# MongoDB connection with proper connection pooling
@lru_cache(maxsize=1)
def get_mongo_client():
    """Create and cache a MongoDB client using environment variables"""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise HTTPException(status_code=500, detail="MongoDB URI not found in environment variables")
    return MongoClient(mongo_uri, maxPoolSize=10)

def get_collection():
    """Get the MongoDB collection"""
    client = get_mongo_client()
    db = client["BMI-Data"]
    return db["patients"]

# Mappings to ensure consistent data processing
TRANSPORT_MAPPING = {
    "Automobile": 0,
    "Bike": 1,
    "Motorbike": 2,
    "Public_Transportation": 3,
    "Walking": 4
}

BINARY_MAPPINGS = {
    'yes': 1, 'no': 0, 'Male': 1, 'Female': 0, 
    True: 1, False: 0, 1: 1, 0: 0
}

ORDINAL_MAPPINGS = {
    'CAEC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'CALC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
}

# Model loading with error handling
def load_model():
    """Load the trained model from disk"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Define request model with proper validation
class PredictionRequest(BaseModel):
    id: int
    Gender: int  # 1 for Male, 0 for Female
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: int  # 1 for yes, 0 for no
    FAVC: int  # Frequent consumption of high caloric food
    FCVC: float  # Frequency of consumption of vegetables
    NCP: float  # Number of main meals
    CAEC: int  # Consumption of food between meals (0-3)
    SMOKE: int  # 1 for yes, 0 for no
    CH2O: float  # Consumption of water daily
    SCC: int  # Calories consumption monitoring
    FAF: float  # Physical activity frequency
    TUE: float  # Time using technology devices
    CALC: int  # Consumption of alcohol (0-3)
    MTRANS: str  # Transportation used

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionRequest):
    """
    Predict BMI class based on the provided health data
    """
    try:
        model = load_model()
        
        # Convert to DataFrame
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Handle MTRANS conversion - simplify by using direct mapping
        if "MTRANS" in input_df.columns:
            # Create one-hot encoded columns for MTRANS
            for transport_type in TRANSPORT_MAPPING.keys():
                col_name = f"MTRANS_{transport_type}"
                input_df[col_name] = 0
                if input_df["MTRANS"].iloc[0] == transport_type:
                    input_df[col_name] = 1
            
            # Remove original MTRANS column
            input_df = input_df.drop(columns=["MTRANS"])
            
        # Ensure all expected model features are present
        for feature in model.feature_names_in_:
            if feature not in input_df.columns:
                input_df[feature] = 0
                
        # Keep only the features used by the model
        input_df = input_df[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities) * 100)
        
        return {
            "bmi_class": BMI_CLASSES[prediction],
            "class_number": int(prediction),
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Simplified Upload endpoint 
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload CSV data to MongoDB with proper validation
    """
    try:
        # Read CSV data
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate required columns
        required_columns = ["Gender", "Age", "Height", "Weight", "NObeyesdad"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}")
        
        # Process binary columns consistently
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: BINARY_MAPPINGS.get(x, 0))
        
        # Process ordinal columns
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Handle target variable
        target_mapping = {v: k for k, v in BMI_CLASSES.items()}
        if "NObeyesdad" in df.columns and df["NObeyesdad"].dtype == 'object':
            df["NObeyesdad"] = df["NObeyesdad"].map(target_mapping)
        
        # Insert data to MongoDB
        collection = get_collection()
        records = df.to_dict(orient="records")
        collection.insert_many(records)
        
        return {"message": f"Successfully uploaded {len(records)} records"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# Feature importance endpoint
@app.get("/feature_importance")
def get_feature_importance():
    """
    Get the importance of each feature in the model
    """
    try:
        model = load_model()
        importance = model.feature_importances_
        feature_names = model.feature_names_in_
        
        # Sort features by importance
        importance_dict = {name: float(importance[i]) for i, name in enumerate(feature_names)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {"feature_importance": sorted_importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

# Simplified retrain endpoint
@app.post("/retrain")
def retrain():
    """
    Retrain the model with data from MongoDB
    """
    try:
        # Get data from MongoDB
        collection = get_collection()
        data = list(collection.find({}, {"_id": 0}))
        
        if not data:
            return {"message": "No data available for retraining"}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Validate target column
        if "NObeyesdad" not in df.columns:
            raise HTTPException(status_code=400, detail="Missing target column 'NObeyesdad'")
        
        # Prepare data
        X = df.drop(columns=["NObeyesdad"], errors='ignore')
        y = df["NObeyesdad"].astype(int)
        
        # Split data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model with optimized parameters
        model = RandomForestClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        
        return {"message": "Model retrained successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

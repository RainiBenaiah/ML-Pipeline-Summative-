from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
import logging
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app and load environment variables
app = FastAPI()
load_dotenv()

# Use environment port or default to 8000
PORT = int(os.getenv("PORT", 8000))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = "models/rf_model.pkl"
BMI_CLASSES = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# MongoDB connection with memory optimization
@lru_cache(maxsize=1)
def get_mongo_client():
    """Create and cache a MongoDB client"""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        logger.error("MONGO_URI environment variable not set")
        raise HTTPException(status_code=500, detail="Database configuration missing")
    return MongoClient(mongo_uri, maxPoolSize=5, connectTimeoutMS=5000, serverSelectionTimeoutMS=5000)

def get_collection():
    """Get the MongoDB collection with error handling"""
    try:
        client = get_mongo_client()
        db = client["BMI-Data"]
        return db["patients"]
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise HTTPException(status_code=503, detail="Database connection failed")
    
@app.get("/")
def root():
    """Root endpoint for quick health checks"""
    return {
        "status": "online",
        "service": "BMI Prediction API",
        "version": "1.0",
        "endpoints": ["/predict", "/upload", "/retrain", "/feature_importance", "/health", "/db-check"]
    }

# Health check endpoint (crucial for Render)
@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "ok"}

# Database check endpoint
@app.get("/db-check")
def db_check():
    """Verify MongoDB connection"""
    try:
        client = get_mongo_client()
        client.admin.command('ping')
        return {"status": "ok", "message": "Database connection successful"}
    except Exception as e:
        logger.error(f"Database check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")

# Model check endpoint
@app.get("/model-check")
def model_check():
    """Verify model exists and can be loaded"""
    try:
        if os.path.exists(MODEL_PATH):
            # Don't actually load the model, just check if file exists
            return {"status": "ok", "message": "Model file exists", "path": MODEL_PATH}
        else:
            return {"status": "warning", "message": "Model file not found", "path": MODEL_PATH}
    except Exception as e:
        logger.error(f"Model check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model check failed: {str(e)}")

# Load model with memory optimization
def load_model():
    """Load the model with memory constraints in mind"""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}")
            return None
        return joblib.load(MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Request model
class PredictionRequest(BaseModel):
    id: int
    Gender: int
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: int
    FAVC: int
    FCVC: float
    NCP: float
    CAEC: int
    SMOKE: int
    CH2O: float
    SCC: int
    FAF: float
    TUE: float
    CALC: int
    MTRANS: str

# Prediction endpoint (memory optimized)
@app.post("/predict")
def predict(data: PredictionRequest):
    """Predict BMI class"""
    try:
        model = load_model()
        if model is None:
            return {"message": "Model not loaded. Please retrain or upload a model."}
        
        # Convert to DataFrame
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Create one-hot encoded columns for MTRANS
        mtrans_value = input_df["MTRANS"].iloc[0]
        transport_options = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
        for option in transport_options:
            input_df[f"MTRANS_{option}"] = 1 if mtrans_value == option else 0
        
        # Drop original MTRANS
        input_df = input_df.drop(columns=["MTRANS"])
        
        # Ensure all model features exist
        for feature in model.feature_names_in_:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Keep only model features
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
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Background task for retraining (to avoid timeout)
def background_retrain():
    """Background task for model retraining"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        logger.info("Starting background retraining task")
        
        # Get MongoDB data
        collection = get_collection()
        data = list(collection.find({}, {"_id": 0}))
        
        if not data:
            logger.warning("No data available for retraining")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        if "NObeyesdad" not in df.columns:
            logger.error("Missing target column 'NObeyesdad'")
            return
        
        # Prepare data
        X = df.drop(columns=["NObeyesdad"], errors='ignore')
        y = df["NObeyesdad"].astype(int)
        
        # Split data
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model (with reduced complexity for memory constraints)
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced significantly for memory
            max_depth=10, 
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=2  # Limit parallel jobs to avoid memory issues
        )
        
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH, compress=3)  # Use compression
        
        logger.info("Model retrained successfully")
    except Exception as e:
        logger.error(f"Background retraining error: {str(e)}")

# Retrain endpoint using background tasks
@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """
    Start retraining in the background to avoid timeouts
    """
    background_tasks.add_task(background_retrain)
    return {"message": "Retraining started in background. This may take a few minutes."}

# Feature importance endpoint
@app.get("/feature_importance")
def get_feature_importance():
    """Get feature importance"""
    try:
        model = load_model()
        if model is None:
            return {"message": "Model not loaded. Please retrain or upload a model."}
            
        importance = model.feature_importances_
        feature_names = model.feature_names_in_
        
        # Get top 10 features to limit response size
        importance_dict = {name: float(importance[i]) for i, name in enumerate(feature_names)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {"feature_importance": sorted_importance}
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

# Simplified upload endpoint
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload CSV data to MongoDB"""
    try:
        # Read CSV in chunks to manage memory
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Process binary columns
        binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'Male': 1, 'Female': 0})
        
        # Process ordinal columns
        ordinal_mappings = {
            'CAEC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
            'CALC': {'Never': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        }
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Handle target variable
        target_mapping = {v: k for k, v in BMI_CLASSES.items()}
        if "NObeyesdad" in df.columns and df["NObeyesdad"].dtype == 'object':
            df["NObeyesdad"] = df["NObeyesdad"].map(target_mapping)
        
        # Insert data in smaller batches to avoid memory issues
        collection = get_collection()
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            records = batch.to_dict(orient="records")
            collection.insert_many(records)
        
        return {"message": f"Successfully uploaded {len(df)} records"}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

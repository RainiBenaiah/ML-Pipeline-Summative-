import pickle
import pandas as pd
import numpy as np
from pymongo import MongoClient
from preprocessing import preprocess_data

# Load trained model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Fetch data from MongoDB
def fetch_data_from_mongo(uri, db_name, collection_name, query={}):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data = pd.DataFrame(list(collection.find(query)))
    return data

# Make predictions
def make_predictions(model, data):
    processed_data = preprocess_data(data)
    predictions = model.predict(processed_data)
    return predictions

if __name__ == "__main__":
    MODEL_PATH = "../models/rfmodel.pkl"
    MONGO_URI = "your_mongodb_uri"
    DB_NAME = "your_db_name"
    COLLECTION_NAME = "your_collection_name"
    
    model = load_model(MODEL_PATH)
    raw_data = fetch_data_from_mongo(MONGO_URI, DB_NAME, COLLECTION_NAME)
    predictions = make_predictions(model, raw_data)
    
    print("Predictions:", predictions)

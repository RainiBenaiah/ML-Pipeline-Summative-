import pickle
import numpy as np
from preprocessing import preprocess_data

def load_model(model_path="models/rfmodel.pkl"):
    """Load the trained Random Forest model from a file."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def predict(model, input_data):
    """Make predictions using the trained model."""
    # Ensure input data is preprocessed
    processed_data = preprocess_data(input_data)
    
    # Convert to numpy array if not already
    if not isinstance(processed_data, np.ndarray):
        processed_data = np.array(processed_data)
    
    # Make predictions
    predictions = model.predict(processed_data)
    return predictions

if __name__ == "__main__":
    model = load_model()
    sample_data = [{"Height": 170, "Weight": 75, "Age": 30, "Gender": "Male"}]  # Example input
    prediction = predict(model, sample_data)
    print("Predicted Class:", prediction)

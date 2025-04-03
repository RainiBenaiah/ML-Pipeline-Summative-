# ML-Pipeline-Summative-

# BMI Classification Project

## Project Overview
This project implements a machine learning pipeline for BMI classification using a Random Forest model. The system classifies individuals into different BMI categories based on input features and provides confidence scores for predictions. The project includes data preprocessing, model training, evaluation, prediction, and a web interface.

## Project Architecture
```
BMI-Classification/
│── README.md
│── notebook/
│   └── bmi_classification.ipynb
│── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│── data/
│   ├── train/
│   └── test/
│── models/
│   └── rf_model.pkl
└── bmiapp/
    └── [React application files]
```

## ML Pipeline Overview

### Data Preprocessing (`preprocessing.py`)
The preprocessing module handles data cleaning, feature engineering, and preparation for model training:

- **Data Cleaning**: Handles missing values, removes outliers, and normalizes data
- **Feature Engineering**: Creates relevant features from raw input data
- **Data Transformation**: Scales numerical features and encodes categorical variables
- **Data Splitting**: Splits data into training and testing sets

Example usage:
```python
from src.preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data(data_path)
```

### Model Training and Evaluation (`model.py`)
The model module handles Random Forest model training, hyperparameter tuning, and evaluation:

- **Model Configuration**: Sets up the Random Forest classifier with appropriate parameters
- **Training**: Fits the model on preprocessed training data
- **Evaluation**: Calculates precision, recall, and F1-score metrics
- **Model Persistence**: Saves and loads trained models

Example usage:
```python
from src.model import train_model, evaluate_model, save_model

# Train the model
rf_model = train_model(X_train, y_train)

# Evaluate the model
metrics = evaluate_model(rf_model, X_test, y_test)

# Save the model
save_model(rf_model, 'models/rf_model.pkl')
```

### Prediction (`prediction.py`)
The prediction module handles making predictions with the trained model:

- **Model Loading**: Loads the trained Random Forest model
- **Prediction**: Processes input data and makes class predictions
- **Confidence Calculation**: Provides confidence scores for predictions

Example usage:
```python
from src.prediction import load_model, predict_bmi_class

# Load the model
model = load_model('models/rf_model.pkl')

# Make a prediction
predicted_class, confidence = predict_bmi_class(model, input_data)
print(f"Predicted BMI Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

## Model Performance

### Classification Report
```
              precision    recall  f1-score   support
           0       0.94      0.92      0.93       379
           1       0.82      0.90      0.86       466
           2       0.79      0.73      0.76       345
           3       0.80      0.80      0.80       386
           4       0.89      0.86      0.88       436
           5       0.97      0.98      0.98       496
           6       1.00      1.00      1.00       606
```

BMI Classes:
- 0: Underweight
- 1: Normal Weight
- 2: Overweight
- 3: Obesity Class I
- 4: Obesity Class II
- 5: Obesity Class III
- 6: Insufficient Weight

### Sample Prediction
```
Model: rf_model.pkl
Predicted BMI Class: Insufficient_Weight
Confidence: 52.96%
```

## Web Application

The project includes a web application built with React and Bootstrap that provides a user-friendly interface for interacting with the BMI classification model. The application is deployed on Vercel.

### Features
- Input form for user data
- BMI classification results with confidence scores
- Visualization of feature importance
- Option to upload new data for retraining

## API Endpoints

The backend API is deployed on Render at `https://ml-pipeline-summative-1q2i.onrender.com/` and provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint, returns API status |
| `/db-check` | GET | Checks database connection status |
| `/model-check` | GET | Verifies model availability and status |
| `/predict` | POST | Accepts input features and returns BMI classification with confidence |
| `/retrain` | POST | Retrains the model with new data |
| `/feature_importance` | GET | Returns feature importance scores from the model |
| `/upload` | POST | Endpoint for uploading new training data |

### Example API Usage

#### Prediction Request
```bash
curl -X POST https://ml-pipeline-summative-1q2i.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": value1, "feature2": value2, ...}'
```

#### Response
```json
{
  "prediction": "Insufficient_Weight",
  "confidence": 52.96,
  "status": "success"
}
```

## Retraining Workflow

The model can be retrained with new data:

1. Collect new data and format it appropriately
2. Upload the data through the web interface or API
3. Initiate retraining process
4. Evaluate the new model performance
5. If performance improves, the new model replaces the old one

## Deployment Architecture

- **ML Backend**: Deployed on Render
- **Web Frontend**: Deployed on Vercel
- **Database**: Connected to the backend for storing historical data and predictions

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- Required Python packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install backend dependencies: `pip install -r requirements.txt`
3. Install frontend dependencies: `cd bmiapp && npm install`

### Running Locally
1. Start the backend: `python app.py`
2. Start the frontend: `cd bmiapp && npm start`

## Future Improvements
- Implement A/B testing for model versions
- Add more advanced feature engineering
- Improve confidence calibration
- Expand the model to cover more health metrics

## Project Demo
For a complete walkthrough of the project and demonstration of its features, watch our demo video:
[BMI Classification Project Demo](https://drive.google.com/file/d/1vwq1lKYboXwcYeu6QV6ORLEb_TT5oBxh/view?usp=sharing)

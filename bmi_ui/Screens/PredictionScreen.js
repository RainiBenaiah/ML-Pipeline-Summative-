import React, { useState } from 'react';
import { Container, Card, Row, Col, Alert, Button, Spinner } from 'react-bootstrap';
import Form from '@rjsf/bootstrap-4';
import validator from '@rjsf/validator-ajv8';
import { Link } from 'react-router-dom';

function PredictionScreen() {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Define the JSON schema for your form
  const schema = {
    type: "object",
    required: ["gender", "age", "height", "weight"],
    properties: {
      id: { 
        type: "integer", 
        title: "ID",
        description: "Unique identifier"
      },
      gender: { 
        type: "integer", 
        title: "Gender", 
        enum: [0, 1],
        enumNames: ["Female", "Male"],
        description: "1 for Male, 0 for Female"
      },
      age: { 
        type: "number", 
        title: "Age", 
        minimum: 0,
        maximum: 120,
        description: "Age in years"
      },
      height: { 
        type: "number", 
        title: "Height", 
        minimum: 0,
        description: "Height in cm"
      },
      weight: { 
        type: "number", 
        title: "Weight", 
        minimum: 0,
        description: "Weight in kg"
      },
      family_history_with_overweight: { 
        type: "integer", 
        title: "Family History with Overweight", 
        enum: [0, 1],
        enumNames: ["No", "Yes"],
        description: "1 for yes, 0 for no"
      },
      FAVC: { 
        type: "integer", 
        title: "Frequent Consumption of High Caloric Food", 
        enum: [0, 1],
        enumNames: ["No", "Yes"],
        description: "1 for yes, 0 for no"
      },
      FCVC: { 
        type: "number", 
        title: "Frequency of Consumption of Vegetables", 
        minimum: 0,
        maximum: 3,
        description: "Scale from 0-3"
      },
      NCP: { 
        type: "number", 
        title: "Number of Main Meals", 
        minimum: 1,
        maximum: 5,
        description: "Number of meals per day"
      },
      CAEC: { 
        type: "integer", 
        title: "Consumption of Food Between Meals", 
        enum: [0, 1, 2, 3],
        enumNames: ["Never", "Sometimes", "Frequently", "Always"],
        description: "Scale from 0-3"
      },
      SMOKE: { 
        type: "integer", 
        title: "Smoking", 
        enum: [0, 1],
        enumNames: ["No", "Yes"],
        description: "1 for yes, 0 for no"
      },
      CH2O: { 
        type: "number", 
        title: "Daily Water Consumption", 
        minimum: 0,
        description: "Liters per day"
      },
      SCC: { 
        type: "integer", 
        title: "Calories Consumption Monitoring", 
        enum: [0, 1],
        enumNames: ["No", "Yes"],
        description: "1 for yes, 0 for no"
      },
      FAF: { 
        type: "number", 
        title: "Physical Activity Frequency", 
        minimum: 0,
        maximum: 3,
        description: "Days per week"
      },
      TUE: { 
        type: "number", 
        title: "Time Using Technology Devices", 
        minimum: 0,
        description: "Hours per day"
      },
      CALC: { 
        type: "integer", 
        title: "Alcohol Consumption", 
        enum: [0, 1, 2, 3],
        enumNames: ["Never", "Sometimes", "Frequently", "Always"],
        description: "Scale from 0-3"
      },
      MTRANS: { 
        type: "string", 
        title: "Transportation Used",
        enum: ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
        description: "Primary mode of transportation"
      }
    }
  };

  // UI customization
  const uiSchema = {
    "ui:submitButtonOptions": {
      submitText: "Predict Obesity Classification",
      props: {
        className: "btn btn-primary btn-lg btn-block mt-3"
      }
    },
    id: { 
      "ui:placeholder": "Enter ID number" 
    },
    gender: {
      "ui:widget": "radio",
      "ui:options": { inline: true }
    },
    age: { 
      "ui:placeholder": "Enter age" 
    },
    height: { 
      "ui:placeholder": "Enter height in cm" 
    },
    weight: { 
      "ui:placeholder": "Enter weight in kg" 
    },
    family_history_with_overweight: {
      "ui:widget": "radio",
      "ui:options": { inline: true }
    },
    FAVC: {
      "ui:widget": "radio",
      "ui:options": { inline: true }
    },
    FCVC: {
      "ui:placeholder": "Enter value between 0-3"
    },
    NCP: {
      "ui:placeholder": "Enter number of meals"
    },
    CAEC: {
      "ui:widget": "radio"
    },
    SMOKE: {
      "ui:widget": "radio",
      "ui:options": { inline: true }
    },
    CH2O: {
      "ui:placeholder": "Enter water consumption in liters"
    },
    SCC: {
      "ui:widget": "radio",
      "ui:options": { inline: true }
    },
    FAF: {
      "ui:placeholder": "Enter days per week"
    },
    TUE: {
      "ui:placeholder": "Enter hours per day"
    },
    CALC: {
      "ui:widget": "radio"
    },
    MTRANS: {
      "ui:placeholder": "Select transportation method"
    }
  };

  // Sample obesity classifications for the result display
  const obesityClasses = {
    0: { label: "Insufficient Weight", risk: "Low" },
    1: { label: "Normal Weight", risk: "Low" },
    2: { label: "Overweight Level I", risk: "Enhanced" },
    3: { label: "Overweight Level II", risk: "Medium" },
    4: { label: "Obesity Type I", risk: "High" },
    5: { label: "Obesity Type II", risk: "Very High" },
    6: { label: "Obesity Type III", risk: "Extremely High" }
  };

  const handleSubmit = async ({ formData }) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // TESTING simulate an API call with a setTimeout
      // Replace this with your actual API endpoint call
      setTimeout(() => {
        // Mock response - replace with actual API call
        const mockResult = {
          class: Math.floor(Math.random() * 7), // Random class 0-6 for demo
          probability: Math.random().toFixed(2)
        };
        
        setPrediction(mockResult);
        setIsLoading(false);
      }, 1500);
      
       Uncomment when ready to use real API
      const response = await fetch('https://ml-pipeline-summative-1q2i.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });
      
      if (!response.ok) throw new Error('Prediction failed');
      const result = await response.json();
      setPrediction(result);
      */
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <Container className="py-4">
      <h2 className="text-center mb-4">Obesity Prediction</h2>
      <p className="text-center mb-4">
        Fill out the form below with patient information to predict obesity classification and health risk.
      </p>
      
      <Row>
        <Col lg={prediction ? 8 : 12}>
          <Card className="shadow mb-4">
            <Card.Header className="bg-primary text-white">
              <h4 className="mb-0">Enter Patient Data</h4>
            </Card.Header>
            <Card.Body>
              <Form
                schema={schema}
                uiSchema={uiSchema}
                validator={validator}
                onSubmit={handleSubmit}
                disabled={isLoading}
              />
            </Card.Body>
          </Card>
        </Col>
        
        {prediction && (
          <Col lg={4}>
            <Card className="shadow mb-4 border-success">
              <Card.Header className="bg-success text-white">
                <h4 className="mb-0">Prediction Result</h4>
              </Card.Header>
              <Card.Body>
                <div className="text-center mb-4">
                  <h3>{obesityClasses[prediction.class].label}</h3>
                  <div className="badge bg-warning text-dark fs-5 my-2">
                    {obesityClasses[prediction.class].risk} Risk
                  </div>
                  <p className="mt-3">
                    <strong>Confidence:</strong> {(prediction.probability * 100).toFixed(0)}%
                  </p>
                </div>
                
                <div className="alert alert-info">
                  <h5>Recommendation:</h5>
                  <p>
                    {prediction.class <= 1 ? 
                      "Maintain current lifestyle with regular check-ups." :
                      prediction.class <= 3 ?
                      "Consider modifications to diet and increased physical activity." :
                      "Medical consultation recommended. Follow structured weight management program."
                    }
                  </p>
                </div>
              </Card.Body>
            </Card>
          </Col>
        )}
      </Row>
      
      {isLoading && (
        <div className="text-center py-4">
          <Spinner animation="border" role="status" className="me-2" />
          <span>Processing prediction...</span>
        </div>
      )}
      
      {error && (
        <Alert variant="danger">
          <Alert.Heading>Error</Alert.Heading>
          <p>{error}</p>
        </Alert>
      )}
      
      <div className="d-flex justify-content-between mt-4">
        <Button as={Link} to="/" variant="outline-secondary">
          &laquo; Back to Home
        </Button>
        <Button as={Link} to="/upload" variant="outline-primary">
          Upload Dataset &raquo;
        </Button>
      </div>
    </Container>
  );
}

export default PredictionScreen;

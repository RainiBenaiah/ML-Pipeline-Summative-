// components/LandingPage.js
import React from 'react';
import { Link } from 'react-router-dom';

function LandingPage() {
  return (
    <div className="landing-page">
      <div className="text-center mb-5">
        <h1 className="display-4 fw-bold">TRUTH WEIGHTS</h1>
        <h2 className="display-6 mb-4">Obesity Classification & Risk Assessment</h2>
        
        <div className="row justify-content-center">
          <div className="col-md-8">
            <p className="lead mb-4">
              Understanding your Body Mass Index (BMI) is crucial for assessing potential health risks. 
              Our advanced ML model helps you determine your obesity classification and provides insights 
              into associated health risks.
            </p>
            <Link to="/predict" className="btn btn-primary btn-lg mb-5">
              Get Your Prediction
            </Link>
          </div>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-12">
          <div className="card shadow">
            <div className="card-header bg-primary text-white">
              <h3 className="mb-0">BMI Classifications and Health Risks</h3>
            </div>
            <div className="card-body">
              <p>
                Body Mass Index (BMI) is a simple calculation using a person's height and weight. The formula is 
                BMI = kg/m². The result is used to categorize a person into weight classifications that may indicate 
                health problems.
              </p>
              
              <div className="table-responsive">
                <table className="table table-striped table-hover">
                  <thead className="table-dark">
                    <tr>
                      <th>BMI Category</th>
                      <th>BMI Range</th>
                      <th>Classification</th>
                      <th>Health Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Insufficient Weight</td>
                      <td>Below 18.5</td>
                      <td>Underweight</td>
                      <td>Malnutrition risk, decreased immune function, osteoporosis</td>
                    </tr>
                    <tr>
                      <td>Normal Weight</td>
                      <td>18.5 - 24.9</td>
                      <td>Normal weight</td>
                      <td>Lowest risk of health issues</td>
                    </tr>
                    <tr>
                      <td>Overweight Level I</td>
                      <td>25.0 - 29.9</td>
                      <td>Overweight</td>
                      <td>Moderate risk of developing heart disease, high blood pressure</td>
                    </tr>
                    <tr>
                      <td>Overweight Level II</td>
                      <td>30.0 - 34.9</td>
                      <td>Obesity (Class I)</td>
                      <td>High risk of developing heart disease, high blood pressure, diabetes</td>
                    </tr>
                    <tr>
                      <td>Overweight Level III</td>
                      <td>35.0 - 39.9</td>
                      <td>Obesity (Class II)</td>
                      <td>Very high risk of developing serious health conditions</td>
                    </tr>
                    <tr>
                      <td>Overweight Level IV</td>
                      <td>40.0 and above</td>
                      <td>Obesity (Class III)</td>
                      <td>Extremely high risk of developing life-threatening conditions</td>
                    </tr>
                    <tr>
                      <td>Special Considerations</td>
                      <td>Varies</td>
                      <td>Metabolic Obesity</td>
                      <td>Risk assessment based on factors beyond BMI (visceral fat, etc.)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              
              <div className="alert alert-info mt-3">
                <p className="mb-0">
                  <strong>Note:</strong> BMI is just one indicator of health. Our model considers multiple factors beyond BMI 
                  for a more comprehensive assessment, including lifestyle, dietary habits, and physical activity.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="text-center mt-4">
        <Link to="/predict" className="btn btn-success me-2">Predict Your Classification</Link>
        <Link to="/visualization" className="btn btn-info ms-2">View Visualizations</Link>
        <Link to="/feature-importance" className="btn btn-warning ms-2">Feature Importance</Link>
        <Link to="/upload" className="btn btn-danger ms-2">Upload Data</Link>
        <Link to="/retrain" className="btn btn-secondary ms-2">Retrain Model</Link>
      </div>
    </div>
  );
}

export default LandingPage;
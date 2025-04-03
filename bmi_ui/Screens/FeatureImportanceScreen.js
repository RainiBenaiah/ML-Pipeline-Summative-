// components/FeatureImportanceScreen.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function FeatureImportanceScreen() {
  const [featureData, setFeatureData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchFeatureImportance = async () => {
      try {
        // Replace with your actual API endpoint
        const response = await axios.get('https://your-api-endpoint/feature-importance');
        setFeatureData(response.data);
      } catch (error) {
        console.error('Error fetching feature importance data:', error);
        setError('Failed to load feature importance data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchFeatureImportance();
  }, []);
  
  // Placeholder data in case the API call doesn't work in this demo
  const placeholderData = {
    features: [
      { name: 'Weight', importance: 0.28, description: 'Body weight in kilograms' },
      { name: 'Height', importance: 0.23, description: 'Body height in meters' },
      { name: 'Age', importance: 0.12, description: 'Age in years' },
      { name: 'Family History', importance: 0.09, description: 'Family history of overweight' },
      { name: 'FAVC', importance: 0.07, description: 'Frequent consumption of high caloric food' },
      { name: 'Physical Activity', importance: 0.06, description: 'Frequency of physical activity' },
      { name: 'Water Consumption', importance: 0.05, description: 'Daily water consumption' },
      { name: 'Technology Use', importance: 0.04, description: 'Time using technology devices' },
      { name: 'Vegetable Consumption', importance: 0.03, description: 'Frequency of vegetable consumption' },
      { name: 'Transportation', importance: 0.02, description: 'Transportation used' },
      { name: 'Gender', importance: 0.01, description: 'Gender (male or female)' }
    ],
    model_info: {
      name: 'XGBoost Classifier',
      date_trained: '2025-03-15',
      accuracy: 0.87
    }
  };
  
  // Use placeholder data for the demo
  const displayData = featureData || placeholderData;
  
  const getImportanceBarStyle = (importance) => {
    // Determine color based on importance value
    let color;
    if (importance > 0.15) {
      color = 'bg-danger'; // High importance
    } else if (importance > 0.05) {
      color = 'bg-warning'; // Medium importance
    } else {
      color = 'bg-success'; // Low importance
    }
    
    return {
      width: `${importance * 100}%`,
      className: `progress-bar ${color}`
    };
  };
  
  return (
    <div className="feature-importance-screen">
      <h2 className="text-center mb-4">Feature Importance Analysis</h2>
      
      {isLoading ? (
        <div className="text-center p-5">
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
          <p className="mt-3">Loading feature importance data...</p>
        </div>
      ) : error ? (
        <div className="alert alert-danger">
          <p className="mb-0">{error}</p>
        </div>
      ) : (
        <div className="row">
          <div className="col-lg-8">
            <div className="card shadow mb-4">
              <div className="card-header bg-primary text-white">
                <h4 className="mb-0">Feature Importance Rankings</h4>
              </div>
              <div className="card-body">
                <div className="alert alert-info mb-4">
                  <p className="mb-0">
                    <strong>What is Feature Importance?</strong> Feature importance indicates how much each feature contributes to the model's predictions. Higher values mean the feature has a stronger influence on obesity classification.
                  </p>
                </div>
                
                <div className="table-responsive">
                  <table className="table table-striped">
                    <thead className="table-dark">
                      <tr>
                        <th style={{ width: '5%' }}>#</th>
                        <th style={{ width: '20%' }}>Feature</th>
                        <th style={{ width: '50%' }}>Importance</th>
                        <th style={{ width: '15%' }}>Value</th>
                        <th style={{ width: '10%' }}>Rank</th>
                      </tr>
                    </thead>
                    <tbody>
                      {displayData.features.map((feature, index) => {
                        const barStyle = getImportanceBarStyle(feature.importance);
                        
                        return (
                          <tr key={index}>
                            <td>{index + 1}</td>
                            <td>{feature.name}</td>
                            <td>
                              <div className="progress">
                                <div 
                                  className={barStyle.className}
                                  role="progressbar" 
                                  style={{ width: barStyle.width }}
                                  aria-valuenow={feature.importance * 100} 
                                  aria-valuemin="0" 
                                  aria-valuemax="100"
                                ></div>
                              </div>
                            </td>
                            <td>{(feature.importance * 100).toFixed(2)}%</td>
                            <td>
                              <span className={`badge ${index < 3 ? 'bg-danger' : index < 7 ? 'bg-warning' : 'bg-success'}`}>
                                {index < 3 ? 'High' : index < 7 ? 'Medium' : 'Low'}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          
          <div className="col-lg-4">
            <div className="card shadow mb-4">
              <div className="card-header bg-info text-white">
                <h4 className="mb-0">Model Information</h4>
              </div>
              <div className="card-body">
                <p><strong>Model Type:</strong> {displayData.model_info.name}</p>
                <p><strong>Last Trained:</strong> {displayData.model_info.date_trained}</p>
                <p><strong>Accuracy:</strong> {(displayData.model_info.accuracy * 100).toFixed(2)}%</p>
                
                <hr />
                
                <h5 className="mt-4">Key Insights</h5>
                <ul className="list-group list-group-flush">
                  <li className="list-group-item">
                    <i className="bi bi-star-fill text-warning me-2"></i>
                    Weight and height are the most influential factors
                  </li>
                  <li className="list-group-item">
                    <i className="bi bi-star-fill text-warning me-2"></i>
                    Family history has significant impact on obesity risk
                  </li>
                  <li className="list-group-item">
                    <i className="bi bi-star-fill text-warning me-2"></i>
                    Dietary habits collectively contribute to over 15% of the model's decisions
                  </li>
                  <li className="list-group-item">
                    <i className="bi bi-star-fill text-warning me-2"></i>
                    Physical activity can offset other risk factors
                  </li>
                </ul>
              </div>
            </div>
            
            <div className="card shadow">
              <div className="card-header bg-success text-white">
                <h4 className="mb-0">Feature Descriptions</h4>
              </div>
              <div className="card-body">
                <div className="accordion" id="featureDescriptions">
                  {displayData.features.slice(0, 5).map((feature, index) => (
                    <div className="accordion-item" key={index}>
                      <h2 className="accordion-header" id={`heading${index}`}>
                        <button 
                          className="accordion-button collapsed" 
                          type="button" 
                          data-bs-toggle="collapse" 
                          data-bs-target={`#collapse${index}`} 
                          aria-expanded="false" 
                          aria-controls={`collapse${index}`}
                        >
                          {feature.name}
                        </button>
                      </h2>
                      <div 
                        id={`collapse${index}`} 
                        className="accordion-collapse collapse" 
                        aria-labelledby={`heading${index}`} 
                        data-bs-parent="#featureDescriptions"
                      >
                        <div className="accordion-body">
                          {feature.description}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default FeatureImportanceScreen;
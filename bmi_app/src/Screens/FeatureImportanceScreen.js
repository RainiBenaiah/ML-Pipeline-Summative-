import React, { useState, useEffect } from 'react';
import axios from 'axios';

function FeatureImportanceScreen() {
  const [featureData, setFeatureData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchFeatureImportance = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/feature_importance');
        setFeatureData(response.data.feature_importance);
      } catch (error) {
        console.error('Error fetching feature importance data:', error);
        setError('Failed to load feature importance data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchFeatureImportance();
  }, []);

  // If the data is available, sort the features by importance in descending order
  const sortedFeatures = featureData
    ? Object.entries(featureData).sort((a, b) => b[1] - a[1])
    : [];

  const getImportanceBarStyle = (importance) => {
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
      className: `progress-bar ${color}`,
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
                        <th style={{ width: '60%' }}>Feature</th>
                        <th style={{ width: '30%' }}>Importance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedFeatures.map(([feature, importance], index) => {
                        const barStyle = getImportanceBarStyle(importance);

                        return (
                          <tr key={index}>
                            <td>{index + 1}</td>
                            <td>{feature}</td>
                            <td>
                              <div className="progress">
                                <div
                                  className={barStyle.className}
                                  role="progressbar"
                                  style={{ width: barStyle.width }}
                                  aria-valuenow={importance * 100}
                                  aria-valuemin="0"
                                  aria-valuemax="100"
                                ></div>
                              </div>
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
        </div>
      )}
    </div>
  );
}

export default FeatureImportanceScreen;


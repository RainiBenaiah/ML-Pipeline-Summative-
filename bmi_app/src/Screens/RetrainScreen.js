// components/RetrainScreen.js
import React, { useState } from 'react';
import axios from 'axios';

function RetrainScreen() {
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainStatus, setRetrainStatus] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  
  const handleRetrain = async () => {
    setIsRetraining(true);
    setRetrainStatus(null);
    setModelComparison(null);
    
    try {
      // Replace with your actual API endpoint
      const response = await axios.post('https://ml-pipeline-summative-1q2i.onrender.com/retrain');
      
      setRetrainStatus({
        success: true,
        message: 'Model retrained successfully!'
      });
      
      setModelComparison(response.data);
    } catch (error) {
      console.error('Error retraining model:', error);
      
      setRetrainStatus({
        success: false,
        message: 'Error retraining model. Please try again.'
      });
    } finally {
      setIsRetraining(false);
    }
  };
  
  const getPerformanceIndicator = (oldMetric, newMetric, higherIsBetter = true) => {
    const diff = newMetric - oldMetric;
    const improved = higherIsBetter ? diff > 0 : diff < 0;
    
    if (Math.abs(diff) < 0.001) {
      return <span className="text-secondary">→ No change</span>;
    }
    
    return improved ? (
      <span className="text-success">
        <i className="bi bi-arrow-up-circle-fill me-1"></i>
        Improved ({Math.abs(diff).toFixed(4)})
      </span>
    ) : (
      <span className="text-danger">
        <i className="bi bi-arrow-down-circle-fill me-1"></i>
        Decreased ({Math.abs(diff).toFixed(4)})
      </span>
    );
  };
  
  return (
    <div className="retrain-screen">
      <h2 className="text-center mb-4">Retrain Model</h2>
      
      <div className="row justify-content-center">
        <div className="col-lg-8">
          <div className="card shadow mb-4">
            <div className="card-header bg-primary text-white">
              <h4 className="mb-0">Model Retraining</h4>
            </div>
            <div className="card-body">
              <div className="alert alert-info mb-4">
                <h5><i className="bi bi-info-circle-fill me-2"></i> About Model Retraining</h5>
                <p>Retraining the model will use the latest available data to update the obesity prediction algorithm.</p>
                <p className="mb-0"><strong>When to retrain:</strong> After adding significant new data or when the model's performance seems to be declining.</p>
              </div>
              
              <div className="d-grid">
                <button
                  className="btn btn-primary btn-lg"
                  onClick={handleRetrain}
                  disabled={isRetraining}
                >
                  {isRetraining ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                      Retraining Model...
                    </>
                  ) : 'Retrain Model'}
                </button>
              </div>
              
              {retrainStatus && (
                <div className={`alert ${retrainStatus.success ? 'alert-success' : 'alert-danger'} mt-4`}>
                  <p className="mb-0">{retrainStatus.message}</p>
                </div>
              )}
            </div>
          </div>
          
          {modelComparison && (
            <div className="card shadow">
              <div className="card-header bg-success text-white">
                <h4 className="mb-0">Model Comparison</h4>
              </div>
              <div className="card-body">
                <h5 className="mb-3">Performance Metrics Comparison</h5>
                
                <div className="table-responsive">
                  <table className="table table-striped table-bordered">
                    <thead className="table-dark">
                      <tr>
                        <th>Metric</th>
                        <th>Previous Model</th>
                        <th>New Model</th>
                        <th>Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Accuracy</td>
                        <td>{modelComparison.old_model.accuracy.toFixed(4)}</td>
                        <td>{modelComparison.new_model.accuracy.toFixed(4)}</td>
                        <td>{getPerformanceIndicator(modelComparison.old_model.accuracy, modelComparison.new_model.accuracy)}</td>
                      </tr>
                      <tr>
                        <td>Precision</td>
                        <td>{modelComparison.old_model.precision.toFixed(4)}</td>
                        <td>{modelComparison.new_model.precision.toFixed(4)}</td>
                        <td>{getPerformanceIndicator(modelComparison.old_model.precision, modelComparison.new_model.precision)}</td>
                      </tr>
                      <tr>
                        <td>Recall</td>
                        <td>{modelComparison.old_model.recall.toFixed(4)}</td>
                        <td>{modelComparison.new_model.recall.toFixed(4)}</td>
                        <td>{getPerformanceIndicator(modelComparison.old_model.recall, modelComparison.new_model.recall)}</td>
                      </tr>
                      <tr>
                        <td>F1 Score</td>
                        <td>{modelComparison.old_model.f1.toFixed(4)}</td>
                        <td>{modelComparison.new_model.f1.toFixed(4)}</td>
                        <td>{getPerformanceIndicator(modelComparison.old_model.f1, modelComparison.new_model.f1)}</td>
                      </tr>
                      <tr>
                        <td>Log Loss</td>
                        <td>{modelComparison.old_model.log_loss.toFixed(4)}</td>
                        <td>{modelComparison.new_model.log_loss.toFixed(4)}</td>
                        <td>{getPerformanceIndicator(modelComparison.old_model.log_loss, modelComparison.new_model.log_loss, false)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-4">
                  <h5>Training Details</h5>
                  <div className="row">
                    <div className="col-md-6">
                      <div className="card bg-light mb-3">
                        <div className="card-body">
                          <h6 className="card-title">Previous Model</h6>
                          <p className="mb-2"><strong>Training Date:</strong> {modelComparison.old_model.training_date}</p>
                          <p className="mb-2"><strong>Samples:</strong> {modelComparison.old_model.samples.toLocaleString()}</p>
                          <p className="mb-0"><strong>Training Time:</strong> {modelComparison.old_model.training_time} seconds</p>
                        </div>
                      </div>
                    </div>
                    <div className="col-md-6">
                      <div className="card bg-light">
                        <div className="card-body">
                          <h6 className="card-title">New Model</h6>
                          <p className="mb-2"><strong>Training Date:</strong> {modelComparison.new_model.training_date}</p>
                          <p className="mb-2"><strong>Samples:</strong> {modelComparison.new_model.samples.toLocaleString()}</p>
                          <p className="mb-0"><strong>Training Time:</strong> {modelComparison.new_model.training_time} seconds</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="alert alert-success mt-4">
                  <h5 className="mb-2">Summary</h5>
                  <p className="mb-0">{modelComparison.summary}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RetrainScreen;

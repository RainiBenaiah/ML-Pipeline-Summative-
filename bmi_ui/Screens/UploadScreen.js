// components/UploadScreen.js
import React, { useState } from 'react';
import axios from 'axios';

function UploadScreen() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadStatus(null);
    setUploadResult(null);
  };
  
  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setUploadStatus({
        success: false,
        message: 'Please select a file to upload.'
      });
      return;
    }
    
    // Check file type (CSV only)
    if (!file.name.endsWith('.csv')) {
      setUploadStatus({
        success: false,
        message: 'Please upload a CSV file.'
      });
      return;
    }
    
    setUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Replace with your actual API endpoint
      const response = await axios.post('https://ml-pipeline-summative-1q2i.onrender.com/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setUploadStatus({
        success: true,
        message: 'File uploaded successfully!'
      });
      
      setUploadResult(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      
      setUploadStatus({
        success: false,
        message: 'Error uploading file. Please try again.'
      });
    } finally {
      setUploading(false);
    }
  };
  
  return (
    <div className="upload-screen">
      <h2 className="text-center mb-4">Upload Data</h2>
      
      <div className="row justify-content-center">
        <div className="col-md-8">
          <div className="card shadow">
            <div className="card-header bg-primary text-white">
              <h4 className="mb-0">Upload Dataset</h4>
            </div>
            <div className="card-body">
              <div className="alert alert-info mb-4">
                <h5><i className="bi bi-info-circle-fill me-2"></i> Instructions</h5>
                <p>Upload a CSV file containing obesity-related data for batch prediction or model evaluation.</p>
                <p className="mb-0"><strong>Required format:</strong> CSV with headers matching the prediction form fields (id, gender, age, height, weight, etc.)</p>
              </div>
              
              <form onSubmit={handleUpload}>
                <div className="mb-4">
                  <label htmlFor="fileUpload" className="form-label">Select CSV File</label>
                  <input
                    type="file"
                    className="form-control"
                    id="fileUpload"
                    accept=".csv"
                    onChange={handleFileChange}
                  />
                  <div className="form-text">File must be in CSV format and include all required features.</div>
                </div>
                
                <div className="d-grid">
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={uploading || !file}
                  >
                    {uploading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        Uploading...
                      </>
                    ) : 'Upload Data'}
                  </button>
                </div>
              </form>
              
              {uploadStatus && (
                <div className={`alert ${uploadStatus.success ? 'alert-success' : 'alert-danger'} mt-4`}>
                  <p className="mb-0">{uploadStatus.message}</p>
                </div>
              )}
              
              {uploadResult && (
                <div className="mt-4">
                  <h5>Upload Results</h5>
                  <div className="table-responsive">
                    <table className="table table-striped table-bordered">
                      <thead className="table-dark">
                        <tr>
                          <th>Metric</th>
                          <th>Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Records Processed</td>
                          <td>{uploadResult.records_processed}</td>
                        </tr>
                        <tr>
                          <td>Valid Records</td>
                          <td>{uploadResult.valid_records}</td>
                        </tr>
                        <tr>
                          <td>Invalid Records</td>
                          <td>{uploadResult.invalid_records}</td>
                        </tr>
                        <tr>
                          <td>Processing Time</td>
                          <td>{uploadResult.processing_time} seconds</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  
                  <div className="mt-3 text-center">
                    <a 
                      href={uploadResult.results_url} 
                      className="btn btn-success" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      download
                    >
                      <i className="bi bi-download me-2"></i>
                      Download Results
                    </a>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadScreen;

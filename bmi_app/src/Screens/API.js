import React, { useState, useEffect } from 'react';
import { Container, Card, Row, Col, Badge, Spinner } from 'react-bootstrap';

function APIStatusPanel() {
  const [apiStatus, setApiStatus] = useState({
    root: { status: 'loading', data: null },
    health: { status: 'loading', data: null },
    dbCheck: { status: 'loading', data: null },
    modelCheck: { status: 'loading', data: null }
  });

  useEffect(() => {
    const fetchEndpointStatus = async (endpoint, key) => {
      try {
        const response = await fetch(`https://ml-pipeline-summative-1q2i.onrender.com${endpoint}`);
        if (!response.ok) {
          throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        setApiStatus(prev => ({
          ...prev,
          [key]: { status: 'success', data }
        }));
      } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        setApiStatus(prev => ({
          ...prev,
          [key]: { status: 'error', error: error.message }
        }));
      }
    };

    // Fetch all endpoints
    fetchEndpointStatus('/', 'root');
    fetchEndpointStatus('/health', 'health');
    fetchEndpointStatus('/db-check', 'dbCheck');
    fetchEndpointStatus('/model-check', 'modelCheck');
  }, []);

  const getStatusBadge = (status) => {
    if (status === 'loading') {
      return <Badge bg="secondary">Checking...</Badge>;
    } else if (status === 'success') {
      return <Badge bg="success">Online</Badge>;
    } else {
      return <Badge bg="danger">Error</Badge>;
    }
  };

  return (
    <Container className="py-3">
      <h3 className="text-center mb-3">API Status</h3>
      <Row>
        {Object.entries(apiStatus).map(([key, { status, data, error }]) => (
          <Col key={key} md={6} lg={3} className="mb-3">
            <Card className="h-100 shadow-sm">
              <Card.Header className={`bg-${status === 'success' ? 'success' : status === 'loading' ? 'secondary' : 'danger'} text-white`}>
                <div className="d-flex justify-content-between align-items-center">
                  <span>{key === 'root' ? 'API' : key === 'dbCheck' ? 'Database' : key === 'modelCheck' ? 'Model' : 'Health'}</span>
                  {getStatusBadge(status)}
                </div>
              </Card.Header>
              <Card.Body className="p-2">
                {status === 'loading' ? (
                  <div className="text-center p-3">
                    <Spinner animation="border" size="sm" className="me-2" />
                    <span>Checking status...</span>
                  </div>
                ) : status === 'error' ? (
                  <div className="text-danger small">{error}</div>
                ) : (
                  <div className="small">
                    {key === 'root' && (
                      <>
                        <p className="mb-1"><strong>Service:</strong> {data.service}</p>
                        <p className="mb-1"><strong>Version:</strong> {data.version}</p>
                        <p className="mb-0"><strong>Endpoints:</strong> {data.endpoints?.length || 0}</p>
                      </>
                    )}
                    {key === 'health' && (
                      <p className="mb-0"><strong>Status:</strong> {data.status}</p>
                    )}
                    {key === 'dbCheck' && (
                      <p className="mb-0"><strong>Message:</strong> {data.message}</p>
                    )}
                    {key === 'modelCheck' && (
                      <>
                        <p className="mb-1"><strong>Status:</strong> {data.status}</p>
                        <p className="mb-0"><strong>Message:</strong> {data.message}</p>
                      </>
                    )}
                  </div>
                )}
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
}

export default APIStatusPanel;

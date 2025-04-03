import React from 'react';
import { Container, Card, Row, Col } from 'react-bootstrap';

function VisualizationScreen() {
  
  const visualizations = [
    {
      id: 1,
      title: 'Correlation Heatmap',
      description: 'This heatmap shows the correlation between different features in our obesity dataset. Darker colors indicate stronger correlations, helping identify which factors tend to occur together.',
      imagePath: '/images/download.png',
      altText: 'Correlation Heatmap'
    },
    {
      id: 2,
      title: 'Log Loss Curve',
      description: 'The log loss curve demonstrates how our model,s performance improved during training. The declining curve shows the reduction in prediction error as the model learns from the training data.',
      imagePath: '/images/rfloglosss',
      altText: 'Log Loss Curve'
    },
    {
      id: 3,
      title: 'Weight vs. Height Distribution',
      description: 'This scatter plot visualizes the relationship between weight and height across different obesity classes. The clustering patterns reveal how these two key metrics help distinguish between classification categories.',
      imagePath: '/images/weightvsheight.png',
      altText: 'Weight vs Height Distribution'
    },
    {
      id: 4,
      title: 'Confusion Matrix',
      description: 'onfusion matrix is a table used to evaluate the performance of the classification model by comparing the bmi classification predicted labels with the actual labels. It consists of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN), helping to measure metrics like accuracy, precision, recall,',
      imagePath: '/images/confusion_matrix.png',
      altText: 'Confusion Matrixs'
    }
  ];
  
  return (
    <Container className="py-5">
      <h2 className="text-center mb-4">Data Visualizations</h2>
      <p className="text-center mb-4">
        These visualizations provide insights into the factors affecting obesity classification and model performance.
      </p>
      
      <Row>
        {visualizations.map((viz) => (
          <Col lg={6} className="mb-4" key={viz.id}>
            <Card className="shadow h-100">
              <Card.Header className="bg-primary text-white">
                <h4 className="mb-0">{viz.title}</h4>
              </Card.Header>
              <Card.Body>
                <div className="text-center mb-3">
                  <img 
                    src={viz.imagePath} 
                    alt={viz.altText} 
                    className="img-fluid rounded" 
                    style={{ maxHeight: '300px' }}
                  />
                </div>
                <Card.Text>{viz.description}</Card.Text>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>

      <div className="text-center mt-4">
        <p className="lead">
          Analyzing these visualizations can help us understand the relationships between different factors
          and their impact on obesity classification.
        </p>
      </div>
    </Container>
  );
}

export default VisualizationScreen;

import React from 'react';
import { Container } from 'react-bootstrap';

function Footer() {
  return (
    <footer className="bg-dark text-white py-4 mt-5">
      <Container>
        <div className="row">
          <div className="col-md-6">
            <h5>TRUTH WEIGHTS</h5>
            <p>Advanced Obesity Classification & Risk Assessment Tool</p>
          </div>
          <div className="col-md-6 text-md-end">
            <p>&copy; {new Date().getFullYear()} TRUTH WEIGHTS. All rights reserved.</p>
            <p className="mb-0">Providing accurate obesity classification for better health outcomes</p>
          </div>
        </div>
      </Container>
    </footer>
  );
}

export default Footer;
// src/components/RecommendationsDisplay.js
import React from 'react';
import { Card, Alert } from 'react-bootstrap';
import { AlertTriangle } from 'lucide-react';

const RecommendationsDisplay = ({ recommendations }) => {
    if (!recommendations || recommendations.length === 0) {
        return (
            <Alert variant="success" className="mt-4">
                ✅ <strong>No significant biases detected!</strong> No specific mitigation strategies are recommended based on the defined thresholds.
            </Alert>
        );
    }

    return (
        <Card className="shadow-sm border-0 mt-4">
            <Card.Header className="bg-light border-0 d-flex align-items-center">
                <AlertTriangle size={20} className="me-2 text-warning" />
                <h5 className="mb-0">Mitigation Recommendations</h5>
            </Card.Header>
            <Card.Body>
                {recommendations.map((rec, index) => (
                    <div key={index} className="mb-3 p-3 border-start border-4 border-warning bg-light rounded">
                        <h6 className="mb-1">{rec.metric}: <span className="text-muted fw-normal">{rec.issue}</span></h6>
                        <strong className="small text-secondary">Suggested Strategies:</strong>
                        <ul className="mt-1 mb-0 ps-4">
                            {rec.strategies.map((strat, i) => <li key={i} className="small">{strat}</li>)}
                        </ul>
                    </div>
                ))}
            </Card.Body>
        </Card>
    );
};

export default RecommendationsDisplay;
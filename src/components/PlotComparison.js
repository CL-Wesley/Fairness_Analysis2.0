// src/components/PlotComparison.js
import React, { useState } from 'react';
import { Row, Col, Card, Nav, Alert, Badge } from 'react-bootstrap';
import { TrendingUp, BarChart3, CheckCircle, XCircle } from 'lucide-react';

const PlotComparison = ({ plotsBefore, plotsAfter }) => {
    const plotKeys = [
        { key: 'accuracy_plot', title: 'Accuracy', icon: TrendingUp },
        { key: 'selection_rate_plot', title: 'Selection Rate', icon: BarChart3 },
        { key: 'true_positive_rate_plot', title: 'True Positive Rate', icon: CheckCircle },
        { key: 'false_positive_rate_plot', title: 'False Positive Rate', icon: XCircle },
    ];

    const availablePlots = plotKeys.filter(plot => plotsBefore[plot.key] && plotsAfter[plot.key]);
    const [selectedPlot, setSelectedPlot] = useState(availablePlots.length > 0 ? availablePlots[0].key : null);

    if (!selectedPlot) {
        return <Alert variant="info">No visualization data is available.</Alert>;
    }

    const selectedPlotInfo = availablePlots.find(p => p.key === selectedPlot);

    return (
        <Card className="shadow-sm border-0">
            <Card.Header className="bg-white border-bottom-0 pt-3 px-3">
                <Nav variant="tabs">
                    {availablePlots.map(plot => (
                        <Nav.Item key={plot.key}>
                            <Nav.Link active={selectedPlot === plot.key} onClick={() => setSelectedPlot(plot.key)} className="d-flex align-items-center">
                                <plot.icon size={16} className="me-2" />
                                {plot.title}
                            </Nav.Link>
                        </Nav.Item>
                    ))}
                </Nav>
            </Card.Header>
            <Card.Body className="p-4 bg-light">
                <h5 className="text-center mb-4">{selectedPlotInfo.title} Comparison</h5>
                <Row className="g-4">
                    <Col lg={6} className="text-center">
                        <Badge bg="secondary" className="mb-2">Before Mitigation</Badge>
                        <div className="border rounded p-2 bg-white plot-image-container">
                            <img src={`data:image/png;base64,${plotsBefore[selectedPlot]}`} alt={`${selectedPlot} Before`} />
                        </div>
                    </Col>
                    <Col lg={6} className="text-center">
                        <Badge bg="success" className="mb-2">After Mitigation</Badge>
                        <div className="border rounded p-2 bg-white plot-image-container">
                            <img src={`data:image/png;base64,${plotsAfter[selectedPlot]}`} alt={`${selectedPlot} After`} />
                        </div>
                    </Col>
                </Row>
            </Card.Body>
        </Card>
    );
};

export default PlotComparison;
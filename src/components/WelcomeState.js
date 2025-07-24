// src/components/WelcomeState.js
import React from 'react';
import { Card, Row, Col, Button } from 'react-bootstrap';
import { Play, Database, BookOpen, TrendingUp, Shield, Users, BarChart3 } from 'lucide-react';

const WelcomeState = () => {
    const sampleDatasets = [
        {
            name: "Criminal Justice Dataset",
            description: "COMPAS recidivism prediction data",
            icon: Shield,
            features: "Gender, Race, Age",
            rows: "7,214 records"
        },
        {
            name: "Financial Credit Dataset",
            description: "Credit approval predictions",
            icon: TrendingUp,
            features: "Gender, Ethnicity",
            rows: "1,000 records"
        },
        {
            name: "Healthcare Outcomes",
            description: "Medical treatment recommendations",
            icon: Users,
            features: "Age, Gender, Ethnicity",
            rows: "5,000 records"
        }
    ];

    const steps = [
        {
            icon: Database,
            title: "Select Dataset",
            description: "Choose from pre-loaded datasets or upload your own data"
        },
        {
            icon: BarChart3,
            title: "Configure Analysis",
            description: "Define target variables and sensitive features"
        },
        {
            icon: Play,
            title: "Run Analysis",
            description: "Apply mitigation strategies and generate fairness reports"
        }
    ];

    return (
        <div className="welcome-container">
            {/* Hero Section */}
            <div className="welcome-hero">
                <div className="hero-content">
                    <h1 className="hero-title">
                        Build Fairer AI Systems with Confidence
                    </h1>
                    <p className="hero-subtitle">
                        Detect, measure, and mitigate algorithmic bias in your machine learning models.
                        Ensure compliance with fairness standards and promote equitable outcomes across all demographic groups.
                    </p>
                    <div className="hero-stats">
                        <div className="stat">
                            <span className="stat-number">99.2%</span>
                            <span className="stat-label">Accuracy</span>
                        </div>
                        <div className="stat">
                            <span className="stat-number">15+</span>
                            <span className="stat-label">Fairness Metrics</span>
                        </div>
                        <div className="stat">
                            <span className="stat-number">3</span>
                            <span className="stat-label">Mitigation Strategies</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Getting Started Steps */}
            <Card className="welcome-section">
                <Card.Header className="section-header">
                    <BookOpen className="section-icon" />
                    <h3>Getting Started</h3>
                </Card.Header>
                <Card.Body>
                    <Row className="g-4">
                        {steps.map((step, index) => (
                            <Col lg={4} key={index}>
                                <div className="step-card">
                                    <div className="step-number">{index + 1}</div>
                                    <step.icon className="step-icon" size={24} />
                                    <h4 className="step-title">{step.title}</h4>
                                    <p className="step-description">{step.description}</p>
                                </div>
                            </Col>
                        ))}
                    </Row>
                </Card.Body>
            </Card>

            {/* Sample Datasets */}
            <Card className="welcome-section">
                <Card.Header className="section-header">
                    <Database className="section-icon" />
                    <h3>Try Sample Datasets</h3>
                    <p className="section-subtitle">Explore pre-loaded datasets to see the platform in action</p>
                </Card.Header>
                <Card.Body>
                    <Row className="g-4">
                        {sampleDatasets.map((dataset, index) => (
                            <Col lg={4} key={index}>
                                <Card className="sample-dataset-card h-100">
                                    <Card.Body className="d-flex flex-column">
                                        <div className="dataset-header">
                                            <dataset.icon className="dataset-icon" size={20} />
                                            <h5 className="dataset-name">{dataset.name}</h5>
                                        </div>
                                        <p className="dataset-description">{dataset.description}</p>
                                        <div className="dataset-meta">
                                            <span className="meta-item">{dataset.features}</span>
                                            <span className="meta-item">{dataset.rows}</span>
                                        </div>
                                        <Button variant="outline-primary" className="mt-auto dataset-button">
                                            <Play size={16} className="me-2" />
                                            Try This Dataset
                                        </Button>
                                    </Card.Body>
                                </Card>
                            </Col>
                        ))}
                    </Row>
                </Card.Body>
            </Card>

            {/* Educational Content */}
            <Card className="welcome-section">
                <Card.Header className="section-header">
                    <Shield className="section-icon" />
                    <h3>Why Fairness Matters</h3>
                </Card.Header>
                <Card.Body>
                    <Row className="g-4">
                        <Col lg={6}>
                            <div className="educational-content">
                                <h4>Regulatory Compliance</h4>
                                <p>Stay ahead of evolving AI regulations including the EU AI Act, NYC Local Law 144, and emerging federal guidelines.</p>
                            </div>
                        </Col>
                        <Col lg={6}>
                            <div className="educational-content">
                                <h4>Risk Mitigation</h4>
                                <p>Proactively identify and address bias before deployment to avoid reputational damage and legal liability.</p>
                            </div>
                        </Col>
                        <Col lg={6}>
                            <div className="educational-content">
                                <h4>Business Value</h4>
                                <p>Fair AI systems lead to better outcomes, increased customer trust, and more inclusive products and services.</p>
                            </div>
                        </Col>
                        <Col lg={6}>
                            <div className="educational-content">
                                <h4>Technical Excellence</h4>
                                <p>Comprehensive metrics and visualizations help you understand model behavior across all demographic groups.</p>
                            </div>
                        </Col>
                    </Row>
                </Card.Body>
            </Card>
        </div>
    );
};

export default WelcomeState;

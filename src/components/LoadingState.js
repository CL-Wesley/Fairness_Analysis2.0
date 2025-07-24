// src/components/LoadingState.js
import React, { useState, useEffect } from 'react';
import { Card, ProgressBar } from 'react-bootstrap';
import { Database, Brain, BarChart3, FileText, CheckCircle, Loader } from 'lucide-react';

const LoadingState = () => {
    const [currentStep, setCurrentStep] = useState(0);
    const [progress, setProgress] = useState(0);

    const steps = [
        {
            icon: Database,
            title: "Loading & Validating Data",
            description: "Analyzing dataset structure and checking data quality",
            duration: 2000
        },
        {
            icon: Brain,
            title: "Model Processing",
            description: "Loading model and computing baseline predictions",
            duration: 3000
        },
        {
            icon: BarChart3,
            title: "Baseline Metrics Calculation",
            description: "Computing fairness metrics before mitigation",
            duration: 2500
        },
        {
            icon: FileText,
            title: "Applying Mitigation Strategy",
            description: "Training improved model with fairness constraints",
            duration: 4000
        },
        {
            icon: BarChart3,
            title: "Generating Visualizations",
            description: "Creating comparison charts and performance plots",
            duration: 1500
        },
        {
            icon: CheckCircle,
            title: "AI Report Generation",
            description: "Analyzing results and generating insights",
            duration: 2000
        }
    ];

    useEffect(() => {
        const totalDuration = steps.reduce((acc, step) => acc + step.duration, 0);
        let elapsed = 0;

        const interval = setInterval(() => {
            elapsed += 100;
            const newProgress = Math.min((elapsed / totalDuration) * 100, 100);
            setProgress(newProgress);

            // Update current step based on elapsed time
            let stepElapsed = 0;
            for (let i = 0; i < steps.length; i++) {
                stepElapsed += steps[i].duration;
                if (elapsed <= stepElapsed) {
                    setCurrentStep(i);
                    break;
                }
            }

            if (elapsed >= totalDuration) {
                clearInterval(interval);
            }
        }, 100);

        return () => clearInterval(interval);
    }, []);

    const getStepStatus = (index) => {
        if (index < currentStep) return 'completed';
        if (index === currentStep) return 'active';
        return 'pending';
    };

    return (
        <div className="loading-container">
            <Card className="loading-card">
                <Card.Body className="text-center">
                    <div className="loading-header">
                        <Loader className="loading-spinner" size={32} />
                        <h2 className="loading-title">Analyzing Your Model for Fairness</h2>
                        <p className="loading-subtitle">
                            This comprehensive analysis examines your model across multiple fairness dimensions
                        </p>
                    </div>

                    <div className="progress-section">
                        <div className="progress-header">
                            <span className="progress-label">Overall Progress</span>
                            <span className="progress-percentage">{Math.round(progress)}%</span>
                        </div>
                        <ProgressBar
                            now={progress}
                            className="main-progress-bar"
                            variant="primary"
                        />
                        <div className="time-estimate">
                            Estimated time remaining: {Math.max(0, Math.round((100 - progress) * 0.15))}s
                        </div>
                    </div>

                    <div className="steps-container">
                        <h4 className="steps-title">Analysis Steps</h4>
                        <div className="steps-list">
                            {steps.map((step, index) => (
                                <div
                                    key={index}
                                    className={`step-item ${getStepStatus(index)}`}
                                >
                                    <div className="step-indicator">
                                        <div className="step-icon-container">
                                            {getStepStatus(index) === 'completed' ? (
                                                <CheckCircle size={20} className="step-icon completed" />
                                            ) : getStepStatus(index) === 'active' ? (
                                                <Loader size={20} className="step-icon active" />
                                            ) : (
                                                <step.icon size={20} className="step-icon pending" />
                                            )}
                                        </div>
                                        {index < steps.length - 1 && (
                                            <div className={`step-connector ${getStepStatus(index) === 'completed' ? 'completed' : ''}`} />
                                        )}
                                    </div>
                                    <div className="step-content">
                                        <h5 className="step-title">{step.title}</h5>
                                        <p className="step-description">{step.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="loading-tips">
                        <h5>Did you know?</h5>
                        <p>Our platform analyzes over 15 different fairness metrics to provide comprehensive bias detection across demographic groups.</p>
                    </div>
                </Card.Body>
            </Card>
        </div>
    );
};

export default LoadingState;

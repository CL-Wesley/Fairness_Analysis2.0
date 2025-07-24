// src/components/MainFairnessCard.js

import React from 'react';
import { Card, Row, Col, Badge, ProgressBar } from 'react-bootstrap';
import { Shield, TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from 'lucide-react';

const MainFairnessCard = ({ overallScores }) => {
    if (!overallScores) {
        return (
            <Card className="main-fairness-card">
                <Card.Body className="text-center">
                    <AlertTriangle size={48} className="text-muted mb-3" />
                    <h5 className="text-muted">Overall fairness data not available</h5>
                </Card.Body>
            </Card>
        );
    }

    const { before_mitigation, after_mitigation, expected_after_mitigation, mitigation_strategy } = overallScores;

    // Calculate improvement
    const actualImprovement = after_mitigation ? after_mitigation.score - before_mitigation.score : 0;
    const expectedImprovement = expected_after_mitigation.score - before_mitigation.score;

    const getScoreIcon = (score) => {
        if (score >= 90) return <CheckCircle size={24} className="text-success" />;
        if (score >= 70) return <Shield size={24} className="text-warning" />;
        return <AlertTriangle size={24} className="text-danger" />;
    };

    const getGradientStyle = (color, score) => ({
        background: `linear-gradient(135deg, ${color}15 0%, ${color}25 100%)`,
        borderLeft: `4px solid ${color}`,
        boxShadow: `0 4px 15px ${color}25`
    });

    const ScoreDisplay = ({ scoreData, title, subtitle, showTrend = false, trendValue = 0 }) => (
        <div className="score-display" style={getGradientStyle(scoreData.color, scoreData.score)}>
            <div className="score-header d-flex justify-content-between align-items-start mb-2">
                <div>
                    <h6 className="score-title mb-1">{title}</h6>
                    <small className="text-muted">{subtitle}</small>
                </div>
                {getScoreIcon(scoreData.score)}
            </div>

            <div className="score-main d-flex align-items-end mb-2">
                <span className="score-number" style={{ color: scoreData.color }}>
                    {scoreData.score}
                </span>
                <span className="score-grade ms-2" style={{ color: scoreData.color }}>
                    {scoreData.grade}
                </span>
                {showTrend && (
                    <div className="trend-indicator ms-2">
                        {trendValue > 0 ? (
                            <TrendingUp size={16} className="text-success" />
                        ) : trendValue < 0 ? (
                            <TrendingDown size={16} className="text-danger" />
                        ) : null}
                        {trendValue !== 0 && (
                            <small className={`ms-1 ${trendValue > 0 ? 'text-success' : 'text-danger'}`}>
                                {trendValue > 0 ? '+' : ''}{trendValue.toFixed(1)}
                            </small>
                        )}
                    </div>
                )}
            </div>

            <ProgressBar
                now={scoreData.score}
                style={{ height: '6px', backgroundColor: `${scoreData.color}20` }}
                className="mb-2"
            >
                <ProgressBar
                    now={scoreData.score}
                    style={{ backgroundColor: scoreData.color }}
                />
            </ProgressBar>

            <small className="score-description text-muted">
                {scoreData.description}
            </small>
        </div>
    );

    return (
        <Card className="main-fairness-card shadow-lg">
            <Card.Header className="bg-light border-0 pb-0">
                <div className="d-flex align-items-center">
                    <Shield size={24} className="text-primary me-2" />
                    <div>
                        <h5 className="mb-0">Overall Fairness Assessment</h5>
                        <small className="text-muted">
                            Comprehensive bias evaluation using {mitigation_strategy} strategy
                        </small>
                    </div>
                </div>
            </Card.Header>

            <Card.Body className="pt-3">
                <Row className="g-3">
                    {/* Before Mitigation */}
                    <Col md={6}>
                        <ScoreDisplay
                            scoreData={before_mitigation}
                            title="Before Mitigation"
                            subtitle="Current model fairness"
                        />
                    </Col>

                    {/* After Mitigation */}
                    <Col md={6}>
                        {after_mitigation ? (
                            <ScoreDisplay
                                scoreData={after_mitigation}
                                title="After Mitigation"
                                subtitle="Improved model fairness"
                                showTrend={true}
                                trendValue={actualImprovement}
                            />
                        ) : (
                            <ScoreDisplay
                                scoreData={expected_after_mitigation}
                                title="Expected After Mitigation"
                                subtitle="Projected improvement"
                                showTrend={true}
                                trendValue={expectedImprovement}
                            />
                        )}
                    </Col>
                </Row>

                {/* Improvement Summary */}
                <div className="improvement-summary mt-4 pt-3 border-top">
                    <Row className="text-center">
                        <Col xs={4}>
                            <div className="summary-stat">
                                <div className="stat-value text-primary">
                                    {after_mitigation ?
                                        (actualImprovement > 0 ? `+${actualImprovement.toFixed(1)}` : actualImprovement.toFixed(1)) :
                                        `+${expectedImprovement.toFixed(1)}`
                                    }
                                </div>
                                <small className="stat-label text-muted">Point Improvement</small>
                            </div>
                        </Col>
                        <Col xs={4}>
                            <div className="summary-stat">
                                <div className="stat-value text-info">
                                    {mitigation_strategy.replace('_', ' ').toUpperCase()}
                                </div>
                                <small className="stat-label text-muted">Strategy Applied</small>
                            </div>
                        </Col>
                        <Col xs={4}>
                            <div className="summary-stat">
                                <div className="stat-value text-success">
                                    {after_mitigation ?
                                        (100 - after_mitigation.improvement_potential).toFixed(0) :
                                        (100 - expected_after_mitigation.improvement_potential).toFixed(0)
                                    }%
                                </div>
                                <small className="stat-label text-muted">Fairness Achieved</small>
                            </div>
                        </Col>
                    </Row>
                </div>

                {/* Action Recommendations */}
                {before_mitigation.score < 70 && (
                    <div className="action-recommendations mt-3 p-3 rounded" style={{ backgroundColor: '#fff3cd' }}>
                        <small className="text-warning">
                            <AlertTriangle size={14} className="me-1" />
                            <strong>Recommendation:</strong> Current fairness score indicates significant bias.
                            Consider implementing additional mitigation strategies or reviewing model features.
                        </small>
                    </div>
                )}

                {(after_mitigation?.score || expected_after_mitigation.score) >= 90 && (
                    <div className="action-recommendations mt-3 p-3 rounded" style={{ backgroundColor: '#d1f2eb' }}>
                        <small className="text-success">
                            <CheckCircle size={14} className="me-1" />
                            <strong>Excellent:</strong> Model demonstrates high fairness standards across all evaluated metrics.
                        </small>
                    </div>
                )}
            </Card.Body>
        </Card>
    );
};

export default MainFairnessCard;

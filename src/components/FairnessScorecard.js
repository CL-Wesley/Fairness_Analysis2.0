// src/components/FairnessScorecard.js
import React from 'react';
import { Card, Col, Row, Badge } from 'react-bootstrap';
import { ArrowUp, ArrowDown, Minus, TrendingUp, TrendingDown, Activity } from 'lucide-react';

const FairnessScorecard = ({ title, metricBefore, metricAfter }) => {
    const valueBefore = metricBefore !== undefined ? parseFloat(metricBefore.toFixed(3)) : 'N/A';
    const valueAfter = metricAfter !== undefined ? parseFloat(metricAfter.toFixed(3)) : 'N/A';

    const isMetricGood = (name, value) => {
        if (value === 'N/A') return false;
        if (name.toLowerCase().includes('disparate impact')) return value >= 0.8 && value <= 1.25;
        return Math.abs(value) <= 0.1;
    };

    const isGoodBefore = isMetricGood(title, valueBefore);
    const isGoodAfter = isMetricGood(title, valueAfter);

    let improvement = 'none';
    let improvementPercentage = 0;

    if (valueBefore !== 'N/A' && valueAfter !== 'N/A') {
        if (title.toLowerCase().includes('disparate impact')) {
            const beforeDistance = Math.abs(1 - valueBefore);
            const afterDistance = Math.abs(1 - valueAfter);
            if (afterDistance < beforeDistance) {
                improvement = 'improved';
                improvementPercentage = Math.round(((beforeDistance - afterDistance) / beforeDistance) * 100);
            } else if (afterDistance > beforeDistance) {
                improvement = 'worsened';
                improvementPercentage = Math.round(((afterDistance - beforeDistance) / beforeDistance) * 100);
            }
        } else {
            const beforeDistance = Math.abs(valueBefore);
            const afterDistance = Math.abs(valueAfter);
            if (afterDistance < beforeDistance) {
                improvement = 'improved';
                improvementPercentage = Math.round(((beforeDistance - afterDistance) / beforeDistance) * 100);
            } else if (afterDistance > beforeDistance) {
                improvement = 'worsened';
                improvementPercentage = Math.round(((afterDistance - beforeDistance) / beforeDistance) * 100);
            }
        }
    }

    const getStatusColor = (isGood) => {
        return isGood ? 'success' : 'danger';
    };

    const getImprovementIcon = () => {
        if (improvement === 'improved') return <TrendingUp size={16} />;
        if (improvement === 'worsened') return <TrendingDown size={16} />;
        return <Activity size={16} />;
    };

    const getImprovementColor = () => {
        if (improvement === 'improved') return 'success';
        if (improvement === 'worsened') return 'danger';
        return 'secondary';
    };

    return (
        <Card className="fairness-scorecard h-100">
            <Card.Body className="d-flex flex-column">
                <div className="scorecard-header">
                    <h5 className="scorecard-title">{title}</h5>
                    <div className="scorecard-status">
                        <Badge
                            bg={getStatusColor(isGoodAfter)}
                            className="status-badge"
                        >
                            {isGoodAfter ? 'FAIR' : 'BIASED'}
                        </Badge>
                    </div>
                </div>

                <div className="scorecard-metrics">
                    <Row className="g-3">
                        <Col xs={6}>
                            <div className="metric-container before">
                                <div className="metric-label">Before</div>
                                <div className="metric-value">
                                    <Badge
                                        pill
                                        bg={getStatusColor(isGoodBefore)}
                                        className="metric-badge"
                                    >
                                        {valueBefore}
                                    </Badge>
                                </div>
                            </div>
                        </Col>
                        <Col xs={6}>
                            <div className="metric-container after">
                                <div className="metric-label">After</div>
                                <div className="metric-value">
                                    <Badge
                                        pill
                                        bg={getStatusColor(isGoodAfter)}
                                        className="metric-badge"
                                    >
                                        {valueAfter}
                                    </Badge>
                                </div>
                            </div>
                        </Col>
                    </Row>
                </div>

                <div className="scorecard-improvement">
                    <div className="improvement-indicator">
                        <div className={`improvement-icon text-${getImprovementColor()}`}>
                            {getImprovementIcon()}
                        </div>
                        <div className="improvement-content">
                            <span className={`improvement-text text-${getImprovementColor()}`}>
                                {improvement === 'improved' && `${improvementPercentage}% Better`}
                                {improvement === 'worsened' && `${improvementPercentage}% Worse`}
                                {improvement === 'none' && 'No Change'}
                            </span>
                            <div className="improvement-description">
                                {improvement === 'improved' && 'Bias reduced successfully'}
                                {improvement === 'worsened' && 'Bias increased after mitigation'}
                                {improvement === 'none' && 'Metrics remained unchanged'}
                            </div>
                        </div>
                    </div>
                </div>
            </Card.Body>
        </Card>
    );
};

export default FairnessScorecard;
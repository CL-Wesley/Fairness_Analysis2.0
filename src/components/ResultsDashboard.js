// src/components/ResultsDashboard.js

import React from 'react';
import { Row, Col, Card, Tab, Nav, Alert, Badge } from 'react-bootstrap';
import { marked } from 'marked';
import { TrendingUp, Shield, Eye, BarChart3, AlertTriangle, GitCompareArrows, Activity, Award } from 'lucide-react';
import FairnessScorecard from './FairnessScorecard';
import MetricComparisonTable from './MetricComparisonTable';
import RecommendationsDisplay from './RecommendationsDisplay';
import PlotComparison from './PlotComparison';
import MainFairnessCard from './MainFairnessCard';

const ResultsDashboard = ({ data }) => {
    if (!data || !data.metrics) {
        return (
            <Alert variant="danger" className="mt-4">
                <AlertTriangle size={20} className="me-2" />
                <strong>Error:</strong> The analysis results are incomplete or malformed.
            </Alert>
        );
    }

    const sensitiveFeature = Object.keys(data.metrics)[0];
    const metricsData = data.metrics[sensitiveFeature];
    const firstComparisonKey = Object.keys(metricsData.fairness_metrics_before)[0];
    const overallMetricsBefore = firstComparisonKey ? metricsData.fairness_metrics_before[firstComparisonKey] : {};
    const overallMetricsAfter = firstComparisonKey ? metricsData.fairness_metrics_after[firstComparisonKey] : {};

    // Extract overall fairness scores from the new API response structure
    const overallFairnessScores = data.overall_fairness_scores || null;

    // Calculate overall fairness improvement (fallback for header display)
    const calculateImprovementScore = () => {
        // Use the new overall fairness scores if available
        if (overallFairnessScores && overallFairnessScores.before_mitigation && overallFairnessScores.after_mitigation) {
            const improvement = overallFairnessScores.after_mitigation.score - overallFairnessScores.before_mitigation.score;
            return Math.max(0, Math.round(improvement));
        }

        // Fallback to old calculation method
        const metrics = [
            { before: overallMetricsBefore.demographic_parity_difference, after: overallMetricsAfter.demographic_parity_difference },
            { before: overallMetricsBefore.equalized_odds_difference, after: overallMetricsAfter.equalized_odds_difference },
            { before: overallMetricsBefore.disparate_impact_ratio, after: overallMetricsAfter.disparate_impact_ratio }
        ];

        let improvementCount = 0;
        let totalMetrics = 0;

        metrics.forEach(metric => {
            if (metric.before !== undefined && metric.after !== undefined) {
                totalMetrics++;
                if (metric.before !== metric.after) {
                    // For disparate impact, closer to 1 is better
                    if (metric === metrics[2]) {
                        if (Math.abs(1 - metric.after) < Math.abs(1 - metric.before)) {
                            improvementCount++;
                        }
                    } else {
                        // For other metrics, closer to 0 is better
                        if (Math.abs(metric.after) < Math.abs(metric.before)) {
                            improvementCount++;
                        }
                    }
                }
            }
        });

        return totalMetrics > 0 ? Math.round((improvementCount / totalMetrics) * 100) : 0;
    };

    const improvementScore = calculateImprovementScore();

    // Use overall fairness score for risk level if available
    const currentScore = overallFairnessScores?.after_mitigation?.score ||
        overallFairnessScores?.before_mitigation?.score ||
        improvementScore;
    const riskLevel = currentScore >= 80 ? 'low' : currentScore >= 60 ? 'medium' : 'high';

    return (
        <div className="results-dashboard">
            {/* Dashboard Header */}
            <div className="dashboard-header">
                <div className="header-content">
                    <div className="header-left">
                        <Shield className="header-icon" size={32} />
                        <div className="header-text">
                            <h1 className="dashboard-title">Fairness Analysis Report</h1>
                            <p className="dashboard-subtitle">
                                Comprehensive bias detection and mitigation analysis
                            </p>
                        </div>
                    </div>
                    <div className="header-right">
                        <div className="summary-stats">
                            <div className="stat-card">
                                <Activity className="stat-icon" size={20} />
                                <div className="stat-content">
                                    <span className="stat-label">Sensitive Feature</span>
                                    <span className="stat-value">{sensitiveFeature}</span>
                                </div>
                            </div>
                            <div className="stat-card">
                                <Award className="stat-icon" size={20} />
                                <div className="stat-content">
                                    <span className="stat-label">Overall Score</span>
                                    <span className="stat-value">
                                        {overallFairnessScores?.after_mitigation?.score ||
                                            overallFairnessScores?.before_mitigation?.score ||
                                            improvementScore}
                                        {overallFairnessScores?.after_mitigation?.grade ||
                                            overallFairnessScores?.before_mitigation?.grade ?
                                            ` (${overallFairnessScores.after_mitigation?.grade || overallFairnessScores.before_mitigation.grade})` :
                                            '%'}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <Badge
                            bg={riskLevel === 'low' ? 'success' : riskLevel === 'medium' ? 'warning' : 'danger'}
                            className="risk-badge"
                        >
                            {riskLevel.toUpperCase()} RISK
                        </Badge>
                    </div>
                </div>
            </div>

            {/* Main Dashboard Content */}
            <Tab.Container id="results-tabs" defaultActiveKey="summary">
                <Card className="dashboard-card">
                    <Card.Header className="dashboard-nav-header">
                        <Nav variant="pills" className="dashboard-nav">
                            <Nav.Item>
                                <Nav.Link eventKey="summary" className="nav-tab">
                                    <TrendingUp size={18} className="nav-icon" />
                                    <span className="nav-label">Executive Summary</span>
                                </Nav.Link>
                            </Nav.Item>
                            <Nav.Item>
                                <Nav.Link eventKey="visuals" className="nav-tab">
                                    <BarChart3 size={18} className="nav-icon" />
                                    <span className="nav-label">Visualizations</span>
                                </Nav.Link>
                            </Nav.Item>
                            <Nav.Item>
                                <Nav.Link eventKey="details" className="nav-tab">
                                    <GitCompareArrows size={18} className="nav-icon" />
                                    <span className="nav-label">Detailed Metrics</span>
                                </Nav.Link>
                            </Nav.Item>
                            <Nav.Item>
                                <Nav.Link eventKey="llm" className="nav-tab">
                                    <Eye size={18} className="nav-icon" />
                                    <span className="nav-label">AI Insights</span>
                                </Nav.Link>
                            </Nav.Item>
                        </Nav>
                    </Card.Header>

                    <Card.Body className="dashboard-content">
                        <Tab.Content>
                            <Tab.Pane eventKey="summary">
                                <div className="summary-pane">
                                    {/* Main Overall Fairness Score Card */}
                                    <div className="main-score-section mb-4">
                                        <MainFairnessCard overallScores={overallFairnessScores} />
                                    </div>

                                    {/* Fairness Scorecard */}
                                    <div className="scorecard-section">
                                        <h3 className="section-title">
                                            <Shield size={24} className="section-icon" />
                                            Detailed Fairness Metrics
                                        </h3>
                                        <Row className="g-4">
                                            <Col lg={4}>
                                                <FairnessScorecard
                                                    title="Demographic Parity"
                                                    metricBefore={overallMetricsBefore.demographic_parity_difference}
                                                    metricAfter={overallMetricsAfter.demographic_parity_difference}
                                                />
                                            </Col>
                                            <Col lg={4}>
                                                <FairnessScorecard
                                                    title="Equalized Odds"
                                                    metricBefore={overallMetricsBefore.equalized_odds_difference}
                                                    metricAfter={overallMetricsAfter.equalized_odds_difference}
                                                />
                                            </Col>
                                            <Col lg={4}>
                                                <FairnessScorecard
                                                    title="Disparate Impact"
                                                    metricBefore={overallMetricsBefore.disparate_impact_ratio}
                                                    metricAfter={overallMetricsAfter.disparate_impact_ratio}
                                                />
                                            </Col>
                                        </Row>
                                    </div>

                                    {/* Key Insights */}
                                    <div className="insights-section">
                                        <h3 className="section-title">
                                            <TrendingUp size={24} className="section-icon" />
                                            Key Insights
                                        </h3>
                                        <div className="insights-grid">
                                            <div className="insight-card">
                                                <h5>Overall Assessment</h5>
                                                <p>
                                                    Your model shows <strong>{improvementScore}% improvement</strong> in fairness metrics
                                                    after applying mitigation strategies. This indicates a{' '}
                                                    <span className={`risk-${riskLevel}`}>{riskLevel} risk</span> of algorithmic bias.
                                                </p>
                                            </div>
                                            <div className="insight-card">
                                                <h5>Compliance Status</h5>
                                                <p>
                                                    {riskLevel === 'low' ? (
                                                        <>Your model <strong>meets</strong> common fairness thresholds and regulatory guidelines.</>
                                                    ) : (
                                                        <>Additional mitigation may be required to meet <strong>regulatory compliance</strong> standards.</>
                                                    )}
                                                </p>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Recommendations */}
                                    <RecommendationsDisplay recommendations={metricsData.recommendations} />
                                </div>
                            </Tab.Pane>

                            <Tab.Pane eventKey="visuals">
                                <div className="visuals-pane">
                                    <h3 className="section-title">
                                        <BarChart3 size={24} className="section-icon" />
                                        Performance Visualizations
                                    </h3>
                                    <PlotComparison
                                        plotsBefore={data.plots.before_mitigation}
                                        plotsAfter={data.plots.after_mitigation}
                                    />
                                </div>
                            </Tab.Pane>

                            <Tab.Pane eventKey="details">
                                <div className="details-pane">
                                    <h3 className="section-title">
                                        <GitCompareArrows size={24} className="section-icon" />
                                        Comprehensive Metrics Analysis
                                    </h3>
                                    <MetricComparisonTable
                                        title="Group Performance Metrics"
                                        metricsBefore={metricsData.group_performance_before}
                                        metricsAfter={metricsData.group_performance_after}
                                    />
                                </div>
                            </Tab.Pane>

                            <Tab.Pane eventKey="llm">
                                <div className="llm-pane">
                                    <h3 className="section-title">
                                        <Eye size={24} className="section-icon" />
                                        AI-Generated Analysis Report
                                    </h3>
                                    <div className="llm-report">
                                        <div
                                            className="prose"
                                            dangerouslySetInnerHTML={{ __html: marked(data.llm_analysis_report) }}
                                        />
                                    </div>
                                </div>
                            </Tab.Pane>
                        </Tab.Content>
                    </Card.Body>
                </Card>
            </Tab.Container>
        </div>
    );
};

export default ResultsDashboard;
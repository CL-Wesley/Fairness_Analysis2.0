// src/components/AnalysisForm.js

import React, { useState, useEffect } from 'react';
import { Form, Button, Spinner, Badge, Alert } from 'react-bootstrap';
import { PlayCircle, UploadCloud, Database, Brain, Settings, CheckCircle } from 'lucide-react';
import { listDatasets, getMitigationStrategies, performAnalysis } from '../api/fairnessAPI';

const AnalysisForm = ({ setIsLoading, setResults, setError, isLoading }) => {
    const [datasets, setDatasets] = useState([]);
    const [strategies, setStrategies] = useState([]);
    const [formData, setFormData] = useState({
        reference_file_url: '',
        target_col: '',
        sensitive_feature: '',
        mitigation_strategy: '',
    });
    const [modelFile, setModelFile] = useState(null);
    const [formProgress, setFormProgress] = useState(0);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const datasetsRes = await listDatasets();
                const availableDatasets = datasetsRes.data.files || [];
                setDatasets(availableDatasets);
                if (availableDatasets.length > 0) {
                    setFormData(prev => ({ ...prev, reference_file_url: availableDatasets[0].url }));
                }

                const strategiesRes = await getMitigationStrategies();
                const availableStrategies = strategiesRes.data.strategies || [];
                setStrategies(availableStrategies);
                if (availableStrategies.length > 0) {
                    setFormData(prev => ({ ...prev, mitigation_strategy: availableStrategies[0].value }));
                }
            } catch (err) {
                setError('Failed to load initial data. Is the backend running?');
            }
        };
        fetchData();
    }, [setError]);

    // Calculate form completion progress
    useEffect(() => {
        const requiredFields = ['reference_file_url', 'target_col', 'sensitive_feature', 'mitigation_strategy'];
        const completedFields = requiredFields.filter(field => formData[field]).length;
        setFormProgress((completedFields / requiredFields.length) * 100);
    }, [formData]);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleFileChange = (e) => {
        setModelFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!formData.reference_file_url || !formData.target_col || !formData.sensitive_feature) {
            setError('Please fill in all required fields: Dataset, Target Column, and Sensitive Feature.');
            return;
        }

        setIsLoading(true);
        setResults(null);
        setError('');

        const submissionData = new FormData();
        Object.keys(formData).forEach(key => submissionData.append(key, formData[key]));
        if (modelFile) {
            submissionData.append('model_file', modelFile);
            submissionData.append('model_format', modelFile.name.split('.').pop());
        }

        try {
            const res = await performAnalysis(submissionData);
            setResults(res.data);
        } catch (err) {
            const errorDetail = err.response?.data?.detail || 'An unknown error occurred. Check browser console and backend logs.';
            setError(`Analysis Failed: ${errorDetail}`);
        } finally {
            setIsLoading(false);
        }
    };

    const getSelectedDataset = () => {
        return datasets.find(d => d.url === formData.reference_file_url);
    };

    const selectedDataset = getSelectedDataset();

    return (
        <div className="analysis-form-container">
            {/* Progress Indicator */}
            <div className="form-progress">
                <div className="progress-header">
                    <span className="progress-label">Configuration Progress</span>
                    <span className="progress-percentage">{Math.round(formProgress)}%</span>
                </div>
                <div className="progress-bar-container">
                    <div
                        className="progress-bar-fill"
                        style={{ width: `${formProgress}%` }}
                    />
                </div>
            </div>

            <Form onSubmit={handleSubmit} className="analysis-form">
                {/* Dataset Selection */}
                <div className="form-section">
                    <div className="section-header">
                        <Database size={18} className="section-icon" />
                        <h4 className="section-title">Dataset Selection</h4>
                        {formData.reference_file_url && <CheckCircle size={16} className="section-check" />}
                    </div>

                    <Form.Group className="form-group">
                        <Form.Label className="form-label">Reference Dataset *</Form.Label>
                        <Form.Select
                            name="reference_file_url"
                            value={formData.reference_file_url}
                            onChange={handleChange}
                            required
                            className="form-input"
                        >
                            <option value="">Select a dataset...</option>
                            {datasets.map((d, i) => (
                                <option key={i} value={d.url}>{d.file_name}</option>
                            ))}
                        </Form.Select>
                        {selectedDataset && (
                            <div className="dataset-preview">
                                <span className="preview-label">Preview:</span>
                                <Badge bg="secondary" className="preview-badge">
                                    {selectedDataset.rows || 'Unknown'} rows
                                </Badge>
                                <Badge bg="secondary" className="preview-badge">
                                    {selectedDataset.columns || 'Unknown'} columns
                                </Badge>
                            </div>
                        )}
                    </Form.Group>
                </div>

                {/* Model Configuration */}
                <div className="form-section">
                    <div className="section-header">
                        <Brain size={18} className="section-icon" />
                        <h4 className="section-title">Model Configuration</h4>
                        {formData.target_col && formData.sensitive_feature && <CheckCircle size={16} className="section-check" />}
                    </div>

                    <Form.Group className="form-group">
                        <Form.Label className="form-label">Target Column *</Form.Label>
                        <Form.Control
                            type="text"
                            name="target_col"
                            placeholder="e.g., two_year_recid"
                            value={formData.target_col}
                            onChange={handleChange}
                            required
                            className="form-input"
                        />
                        <Form.Text className="form-help">
                            The column containing the outcome you want to predict
                        </Form.Text>
                    </Form.Group>

                    <Form.Group className="form-group">
                        <Form.Label className="form-label">Sensitive Feature *</Form.Label>
                        <Form.Control
                            type="text"
                            name="sensitive_feature"
                            placeholder="e.g., race, gender"
                            value={formData.sensitive_feature}
                            onChange={handleChange}
                            required
                            className="form-input"
                        />
                        <Form.Text className="form-help">
                            The demographic attribute to analyze for bias
                        </Form.Text>
                    </Form.Group>

                    <Form.Group className="form-group">
                        <Form.Label className="form-label">
                            <UploadCloud size={16} className="me-2" />
                            Upload Model (Optional)
                        </Form.Label>
                        <Form.Control
                            type="file"
                            onChange={handleFileChange}
                            accept=".onnx,.pkl,.joblib"
                            className="form-input file-input"
                        />
                        {modelFile && (
                            <div className="file-info">
                                <Badge bg="success" className="file-badge">
                                    {modelFile.name}
                                </Badge>
                            </div>
                        )}
                        <Form.Text className="form-help">
                            ONNX format recommended for best compatibility
                        </Form.Text>
                    </Form.Group>
                </div>

                {/* Mitigation Strategy */}
                <div className="form-section">
                    <div className="section-header">
                        <Settings size={18} className="section-icon" />
                        <h4 className="section-title">Mitigation Strategy</h4>
                        {formData.mitigation_strategy && <CheckCircle size={16} className="section-check" />}
                    </div>

                    <Form.Group className="form-group">
                        <Form.Label className="form-label">Strategy *</Form.Label>
                        <Form.Select
                            name="mitigation_strategy"
                            value={formData.mitigation_strategy}
                            onChange={handleChange}
                            required
                            className="form-input"
                        >
                            <option value="">Select strategy...</option>
                            {strategies.map(s => (
                                <option key={s.value} value={s.value}>{s.label}</option>
                            ))}
                        </Form.Select>
                        <Form.Text className="form-help">
                            Choose how to address detected bias in your model
                        </Form.Text>
                    </Form.Group>
                </div>

                {/* Submit Button */}
                <div className="form-actions">
                    <Button
                        variant="primary"
                        type="submit"
                        disabled={isLoading || formProgress < 100}
                        className="submit-button"
                        size="lg"
                    >
                        {isLoading ? (
                            <>
                                <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" />
                                <span className="ms-2">Analyzing...</span>
                            </>
                        ) : (
                            <>
                                <PlayCircle size={20} className="me-2" />
                                Run Fairness Analysis
                            </>
                        )}
                    </Button>

                    {formProgress < 100 && (
                        <Alert variant="info" className="requirements-alert">
                            <strong>Required:</strong> Please complete all required fields to run analysis.
                        </Alert>
                    )}
                </div>
            </Form>
        </div>
    );
};

export default AnalysisForm;

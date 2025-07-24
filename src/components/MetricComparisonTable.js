// src/components/MetricComparisonTable.js
import React from 'react';
import { Table, Card } from 'react-bootstrap';

const MetricComparisonTable = ({ title, metricsBefore, metricsAfter }) => {
    const groupNames = [...new Set([...Object.keys(metricsBefore), ...Object.keys(metricsAfter)])];
    if (groupNames.length === 0) return null;

    const metricKeys = Object.keys(metricsBefore[groupNames[0]] || metricsAfter[groupNames[0]] || {});

    return (
        <Card className="shadow-sm border-0">
            <Card.Header className="bg-light border-0"><h5 className="mb-0">{title}</h5></Card.Header>
            <Card.Body className="p-0">
                <Table responsive striped bordered hover className="align-middle mb-0">
                    <thead className="table-light">
                        <tr>
                            <th rowSpan="2" className="align-middle text-center">Group</th>
                            {metricKeys.map(key => <th key={key} colSpan="2" className="text-center">{key.replace(/_/g, ' ')}</th>)}
                        </tr>
                        <tr>
                            {metricKeys.map(key => (
                                <React.Fragment key={key}>
                                    <th className="text-center small text-muted">Before</th>
                                    <th className="text-center small text-muted">After</th>
                                </React.Fragment>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {groupNames.map(group => (
                            <tr key={group}>
                                <td className="fw-bold text-center">{group}</td>
                                {metricKeys.map(key => {
                                    const valBefore = metricsBefore[group]?.[key];
                                    const valAfter = metricsAfter[group]?.[key];
                                    const improved = valBefore !== undefined && valAfter !== undefined && valAfter > valBefore;
                                    return (
                                        <React.Fragment key={key}>
                                            <td className="text-center">{valBefore !== undefined ? valBefore.toFixed(4) : 'N/A'}</td>
                                            <td className={`text-center fw-bold ${improved ? 'text-success' : 'text-danger'}`}>{valAfter !== undefined ? valAfter.toFixed(4) : 'N/A'}</td>
                                        </React.Fragment>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </Table>
            </Card.Body>
        </Card>
    );
};

export default MetricComparisonTable;
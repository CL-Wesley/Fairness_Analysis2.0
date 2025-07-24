import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000'; // Your backend URL

const api = axios.create({
  baseURL: API_URL,
});

export const listDatasets = () => api.get('/list_datasets');

export const getMitigationStrategies = () => api.get('/mitigation_strategies');

export const performAnalysis = (formData) => {
  return api.post('/perform_analysis', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};
# functions.py
"""
Core utility functions for fairness analysis backend. (V2 - Secure & Robust)
Handles secure model loading, data fetching, preprocessing, and LLM integration.
"""
import sys
from fastapi import HTTPException, UploadFile
import joblib
import pickle
import pandas as pd
import numpy as np
import io
import json
import os
import warnings
import requests
import chardet
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

try:
    import boto3
except ImportError:
    boto3 = None

# --- Import from rewritten fairness_analyzer ---
from fairness_analyzer import FairnessAnalyzer as CoreFairnessAnalyzer

try:
    import onnxruntime as rt
except ImportError:
    rt = None
# --- Load .env from project root if present ---
from pathlib import Path
try:
    from dotenv import load_dotenv
    # Find the project root (parent of Backend)
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
except ImportError:
    # If python-dotenv is not installed, skip loading .env
    pass

# --- CRITICAL SECURITY FIX: Load credentials from environment variables ---
# You MUST set these variables in your deployment environment.
EXTERNAL_S3_API_URL = os.getenv("EXTERNAL_S3_API_URL", "http://xailoadbalancer-579761463.ap-south-1.elb.amazonaws.com/api/files_download/Fairness")
EXTERNAL_S3_ACCESS_TOKEN = os.getenv("EXTERNAL_S3_ACCESS_TOKEN")

if not EXTERNAL_S3_ACCESS_TOKEN:
    # This will prevent the app from starting without the token, which is good practice.
    raise ValueError("FATAL: EXTERNAL_S3_ACCESS_TOKEN environment variable is not set.")

# --- ONNX Model Wrapper (No changes needed, it's solid) ---
class ONNXModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, session: rt.InferenceSession, classes: np.ndarray = np.array([0,1])):
        self.session = session
        self.inputs  = session.get_inputs()
        self.outputs = session.get_outputs()
        self.classes_ = classes
        self._estimator_type = "classifier"

        if len(self.inputs) != 1:
            raise ValueError(f"Expected 1 input, got {len(self.inputs)}.")
        self.input_name = self.inputs[0].name

        self.proba_name = None
        self.label_name = None
        for out in self.outputs:
            if 'proba' in out.name.lower():
                self.proba_name = out.name
            else:
                self.label_name = out.name

        if self.label_name is None:
            raise ValueError("Could not find label output in ONNX model.")

    def fit(self, X, y=None): return self
    def predict(self, X):
        X_onnx = (X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.array(X, dtype=np.float32))
        return self.session.run([self.label_name], {self.input_name: X_onnx})[0]
    def predict_proba(self, X):
        if not self.proba_name:
            # Return a dummy probability array if no proba output, to not break post-processing
            preds = self.predict(X)
            return np.array([[1-p, p] for p in preds], dtype=np.float32)
        X_onnx = (X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.array(X, dtype=np.float32))
        raw = self.session.run([self.proba_name], {self.input_name: X_onnx})[0]
        if isinstance(raw, np.ndarray): return raw
        return np.array([[d.get(c, d.get(str(c))) for c in self.classes_] for d in raw], dtype=float)
    def score(self, X, y): return accuracy_score(y, self.predict(X))

# --- LLM Analysis (No changes needed, but ensure ENV VARS are set) ---
class LLMBasedAnalysis:
    # ... (Your existing LLMBasedAnalysis class code here) ...
    def __init__(self, bedrock_runtime=None):
        if bedrock_runtime is not None:
            self.bedrock_runtime = bedrock_runtime
        else:
            self.bedrock_runtime = None
            self._initialize_bedrock()

    def _initialize_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            # Check for env vars
            if not all(os.getenv(k) for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]):
                warnings.warn("AWS credentials not set in environment variables. LLM analysis will be disabled.")
                self.bedrock_runtime = None
                return

            if boto3 is None:
                warnings.warn("boto3 is not installed. LLM analysis will be disabled.")
                self.bedrock_runtime = None
                return

            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1"), # Use AWS_REGION from .env
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except ImportError:
            warnings.warn("boto3 is not installed. LLM analysis will be disabled.")
            self.bedrock_runtime = None
        except Exception as e:
            warnings.warn(f"Bedrock initialization failed: {e}. LLM analysis will be disabled.")
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str) -> str:
        if not self.bedrock_runtime:
            return "LLM analysis is disabled. Please configure AWS credentials for Bedrock."
        # ... (rest of your _invoke_claude implementation) ...
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 13000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }
            response = self.bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                body=json.dumps(request_body),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["content"][0]["text"]
        except Exception as e:
            return f"Claude invocation failed: {e}"

    def analyze_fairness_disparity(self, summary: str, mitigation_strategy: str = None) -> str:
        # ... (Your existing prompt logic here) ...
        mitigation_info = f"\nMitigation strategy applied: {mitigation_strategy}." if mitigation_strategy else ""
        prompt = f"""
You are an expert in ethical AI, fairness auditing, and responsible machine learning.
Below is a summary of model performance across different demographic groups, generated using fairness metrics like accuracy and selection rate:{mitigation_info}
{summary}
Please analyze and provide the following:
1. **Fairness Disparities**
   - Are there notable disparities in accuracy, selection rate, or error rates across groups?
   - Quantify the disparities clearly.
2. **Potential Harms**
   - What kind of real-world or reputational risks do these disparities pose?
   - Which groups are likely to be disadvantaged?
3. **Compliance & Ethical Implications**
   - Are the disparities likely to violate fairness goals like Demographic Parity or Equal Opportunity?
   - Could this raise regulatory or legal concerns?
4. **Suggested Mitigations**
   - Recommend bias mitigation strategies (e.g., model retraining, fairness constraints)
   - Mention if post-processing or reweighting might help
   - Based on the observed results and the mitigation strategy applied, specify which class of mitigation (pre-processing, in-processing, post-processing) would be most appropriate for further improvement, and why.
5. **Communication to Stakeholders**
   - How would you explain this issue to a business or legal team?
Use markdown formatting with clear section headings and bullet points. Avoid technical jargon where possible.
"""
        return self._invoke_claude(prompt)

# Main orchestrator class
class FairnessPipeline:
    def __init__(self):
        self.llm_analyzer = LLMBasedAnalysis()
        self.core_analyzer = CoreFairnessAnalyzer()

    def get_mitigation_strategies(self):
        return [
            {"label": "Exponentiated Gradient (In-processing)", "value": "exponentiated_gradient"},
            {"label": "Grid Search (In-processing)", "value": "grid_search"},
            {"label": "Reweighing (Pre-processing)", "value": "reweighing"},
            {"label": "Threshold Optimizer (Post-processing)", "value": "postprocessing"}
        ]

    def generate_llm_report(self, title: str, data_dict: Dict[str, Any], mitigation_strategy: str = None) -> str:
        # ... (your existing generate_plot_descriptions logic) ...
        summary = ""
        for plot_title, metrics_data in data_dict.items():
            summary += f"--- {plot_title} ---\n"
            if isinstance(metrics_data, dict):
                 for group_name, metric_values in metrics_data.items():
                    summary += f"Group/Comparison: {group_name}\n"
                    if isinstance(metric_values, dict):
                        for metric, value in metric_values.items():
                            summary += f"  {metric}: {value:.4f}\n"
                    else:
                        summary += f"  Value: {metric_values}\n"
            else:
                summary += f"  Value: {metrics_data}\n"
            summary += "\n"
        return self.llm_analyzer.analyze_fairness_disparity(summary, mitigation_strategy)

    def get_s3_file_metadata(self):
        headers = {"Authorization": f"Bearer {EXTERNAL_S3_ACCESS_TOKEN}"}
        try:
            response = requests.get(EXTERNAL_S3_API_URL, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json().get("files", [])
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Could not connect to external S3 API: {e}")

    def download_and_load_dataframe(self, file_url: str) -> pd.DataFrame:
        try:
            response = requests.get(file_url, timeout=10)
            response.raise_for_status()
            file_content = response.content
            
            # Robust CSV loading
            result = chardet.detect(file_content)
            encoding = result['encoding'] or 'utf-8'
            for delimiter in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, delimiter=delimiter)
                    if len(df.columns) > 1: return df
                except Exception:
                    continue
            raise RuntimeError("Failed to parse CSV. Please check the file's delimiter and format.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download file from {file_url}: {e}")

    def load_model_from_url(self, model_url: str, model_format: str) -> object:
        """
        Downloads and loads a model from a URL, prioritizing ONNX and issuing a security warning for pickle.
        """
        try:
            response = requests.get(model_url, timeout=30)
            response.raise_for_status()
            model_content = response.content
            
            if model_format == 'onnx':
                if rt is None:
                    raise HTTPException(status_code=501, detail="ONNX runtime is not installed.")
                try:
                    session = rt.InferenceSession(model_content)
                    return ONNXModelWrapper(session)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid ONNX file. Error: {e}")
            elif model_format in ['pkl', 'joblib']:
                # --- SECURITY FIX: Add explicit warning ---
                warnings.warn(
                    "Loading models with pickle or joblib is insecure and can execute arbitrary code. "
                    "Only load files from a trusted source. Use the ONNX format for better security and reliability.",
                    UserWarning
                )
                try:
                    model = pickle.loads(model_content) if model_format == 'pkl' else joblib.load(io.BytesIO(model_content))
                    if not hasattr(model, 'predict'):
                        raise AttributeError("Model must have a .predict() method.")
                    return model
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to load pkl/joblib file. Error: {e}")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model format '{model_format}'.")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download model from {model_url}: {e}")

    def load_uploaded_model(self, model_file: UploadFile, model_format: str) -> object:
        """
        Loads a model, prioritizing ONNX and issuing a security warning for pickle.
        """
        contents = model_file.file.read()
        if model_format == 'onnx':
            if rt is None:
                raise HTTPException(status_code=501, detail="ONNX runtime is not installed.")
            try:
                session = rt.InferenceSession(contents)
                return ONNXModelWrapper(session)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid ONNX file. Error: {e}")
        elif model_format in ['pkl', 'joblib']:
            # --- SECURITY FIX: Add explicit warning ---
            warnings.warn(
                "Loading models with pickle or joblib is insecure and can execute arbitrary code. "
                "Only load files from a trusted source. Use the ONNX format for better security and reliability.",
                UserWarning
            )
            try:
                model = pickle.loads(contents) if model_format == 'pkl' else joblib.load(io.BytesIO(contents))
                if not hasattr(model, 'predict'):
                    raise AttributeError("Model must have a .predict() method.")
                return model
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load pkl/joblib file. Error: {e}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model format '{model_format}'.")

    def load_and_preprocess_data(self, df: pd.DataFrame, target_col: Optional[str], sensitive_features: List[str]) -> pd.DataFrame:
        """
        Robustly preprocesses DataFrame.
        """
        if df is None: return pd.DataFrame()
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.strip()

        # Input validation
        all_cols = sensitive_features + ([target_col] if target_col else [])
        for col in all_cols:
             if col not in df_copy.columns:
                 raise ValueError(f"Column '{col}' not found in the dataset.")

        categorical_cols = [col for col in df_copy.select_dtypes(include=['object', 'category']).columns if col not in sensitive_features]
        
        for col in categorical_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype(str).str.strip().str.lower()
                binary_map = {'yes': 1, 'no': 0, 'approved': 1, 'rejected': 0}
                unique_vals = set(df_copy[col].dropna().unique())
                if unique_vals.issubset(binary_map.keys()):
                    df_copy[col] = df_copy[col].map(binary_map)
                else:
                    df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))
        
        # Binarize target if needed
        if target_col and target_col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[target_col]):
            unique_vals = df_copy[target_col].unique()
            if len(unique_vals) > 2:
                most_freq = df_copy[target_col].mode()[0]
                df_copy[target_col] = (df_copy[target_col] != most_freq).astype(int)
            elif len(unique_vals) == 2 and not set(unique_vals).issubset({0, 1}):
                df_copy[target_col] = (df_copy[target_col] == max(unique_vals)).astype(int)

        return df_copy
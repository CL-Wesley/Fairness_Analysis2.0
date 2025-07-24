# main.py
"""
FastAPI backend for Fairness Analysis. (V2 - Refactored & Enhanced)
Orchestrates fairness analysis using the FairnessPipeline class.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io
import numpy as np
import base64
import logging
import traceback
from typing import Dict, List, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.base import BaseEstimator
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import matplotlib.pyplot as plt

# Import the new pipeline class and core analyzer from functions.py
from functions import FairnessPipeline, CoreFairnessAnalyzer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fairness Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use dependency injection for our main pipeline class
def get_fairness_pipeline():
    return FairnessPipeline()

# --- Helper Functions ---
def plot_metrics_to_base64(frame: MetricFrame, title_suffix="") -> Dict[str, str]:
    """Generates and encodes plots from a MetricFrame into base64 PNG images."""
    plots = {}
    metrics_to_plot = {
        'accuracy': 'skyblue',
        'selection_rate': 'lightcoral',
        'true_positive_rate': 'lightgreen',
        'false_positive_rate': 'salmon'
    }
    
    for metric, color in metrics_to_plot.items():
        if metric in frame.by_group:
            fig, ax = plt.subplots(figsize=(8, 5))
            try:
                frame.by_group[metric].plot.bar(ax=ax, rot=45, color=color, legend=False)
                ax.set_title(f'{metric.replace("_", " ").title()} by Group {title_suffix}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_xlabel('Group')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                plots[f'{metric}_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to generate plot for metric '{metric}': {e}")
                plt.close(fig)
    return plots

def apply_mitigation(
    model: BaseEstimator, X_train, X_test, y_train, y_test, A_train, A_test,
    strategy: str,
    y_prob_test_before: Optional[np.ndarray] = None # Pass initial probabilities for post-processing
) -> np.ndarray:
    """
    Applies a specified fairness mitigation strategy and returns mitigated predictions.
    This function now assumes the calling function has validated the strategy.
    """
    logger.info(f"Executing mitigation strategy: '{strategy}'")

    # --- REFACTORED LOGIC ---
    if strategy == "reweighing":
        sens_name = A_train.name or "sensitive_feature"
        train_df = pd.concat([X_train, y_train, A_train], axis=1)
        
        # Make unprivileged/privileged groups dynamic
        unique_groups = sorted(A_train.unique())
        unprivileged_groups=[{sens_name: unique_groups[0]}]
        privileged_groups=[{sens_name: unique_groups[1]}]

        train_bld = BinaryLabelDataset(df=train_df, label_names=[y_train.name], protected_attribute_names=[sens_name])
        reweigher = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        train_bld_transf = reweigher.fit_transform(train_bld)
        
        # Fit the model with sample weights
        model.fit(X_train, y_train, sample_weight=train_bld_transf.instance_weights)
        return model.predict(X_test)

    elif strategy == "exponentiated_gradient":
        mitigator = ExponentiatedGradient(estimator=model, constraints=DemographicParity())
        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        return mitigator.predict(X_test)
        
    elif strategy == "grid_search":
        mitigator = GridSearch(estimator=model, constraints=DemographicParity())
        mitigator.fit(X_train, y_train, sensitive_features=A_train)
        return mitigator.predict(X_test)

    elif strategy == "postprocessing":
        if y_prob_test_before is None:
             raise ValueError("Post-processing requires initial probability predictions.")
        
        # ThresholdOptimizer needs an unfitted estimator if prefit=False, 
        # but here we assume the base model is already decided.
        # We use a 'prefit' approach, assuming the original model is what we want to optimize.
        postproc = ThresholdOptimizer(estimator=model, constraints="demographic_parity", prefit=True)
        
        # Fit the optimizer on the test set predictions
        postproc.fit(X_test, y_test, sensitive_features=A_test, y_pred=y_prob_test_before)
        return postproc.predict(X_test, sensitive_features=A_test)
    
    else:
        raise NotImplementedError(f"Mitigation strategy '{strategy}' is not implemented.")


# --- API Endpoints ---
@app.get("/list_datasets", summary="List available datasets from S3")
async def list_datasets(pipeline: FairnessPipeline = Depends(get_fairness_pipeline)):
    try:
        return {"files": pipeline.get_s3_file_metadata()}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/preview_dataset", summary="Get a preview of a dataset from a URL")
async def preview_dataset(file_url: str, pipeline: FairnessPipeline = Depends(get_fairness_pipeline)):
    try:
        df = pipeline.download_and_load_dataframe(file_url)
        return JSONResponse(content={"columns": df.columns.tolist(), "data": df.head(5).to_dict("records")})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mitigation_strategies", summary="Get available mitigation strategies")
async def get_strategies(pipeline: FairnessPipeline = Depends(get_fairness_pipeline)):
    return {"strategies": pipeline.get_mitigation_strategies()}

@app.post("/perform_analysis", summary="Perform full fairness analysis")
async def perform_analysis(
    reference_file_url: str = Form(...),
    target_col: str = Form(...),
    sensitive_feature: str = Form(...),
    current_file_url: Optional[str] = Form(None),
    mitigation_strategy: str = Form("exponentiated_gradient"),
    model_file: Optional[UploadFile] = File(None),
    model_format: str = Form("onnx"),
    pipeline: FairnessPipeline = Depends(get_fairness_pipeline)
):
    try:
        # 1. Load and Preprocess Data
        logger.info("Loading and preprocessing data...")
        df_ref = pipeline.download_and_load_dataframe(reference_file_url)
        df_proc = pipeline.load_and_preprocess_data(df_ref, target_col, [sensitive_feature])
        
        X = df_proc.drop(columns=[target_col, sensitive_feature])
        y = df_proc[target_col]
        A = df_proc[sensitive_feature]

        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
            X, y, A, test_size=0.3, random_state=42, stratify=y
        )
        
        # 2. Load or Define Model
        is_pretrained = model_file is not None
        if is_pretrained:
            logger.info(f"Loading user-uploaded model (format: {model_format})")
            model = pipeline.load_uploaded_model(model_file, model_format)
        else:
            logger.info("No model uploaded. A default RandomForestClassifier will be used for mitigation.")
            model = RandomForestClassifier(random_state=42)
            # We only fit here if it's not a pre-trained model and no mitigation that requires training is used.
            # Mitigation functions will handle their own fitting.

        # --- FIX: New logic to validate and adjust mitigation strategy ---
        in_processing_strategies = ["exponentiated_gradient", "grid_search", "reweighing"]
        
        if is_pretrained and mitigation_strategy in in_processing_strategies:
            logger.warning(
                f"Incompatible mitigation '{mitigation_strategy}' selected for a pre-trained model. "
                f"Pre-trained models cannot be retrained. "
                f"Switching to 'postprocessing' (ThresholdOptimizer)."
            )
            mitigation_strategy = "postprocessing"
        
        # 3. Evaluate Before Mitigation
        logger.info("Evaluating model before mitigation...")
        # For a fair 'before' comparison, always use the base model
        # If a model was uploaded, use it. If not, train a fresh one for the baseline.
        base_model_for_comparison = model if is_pretrained else RandomForestClassifier(random_state=42)
        if not is_pretrained:
            base_model_for_comparison.fit(X_train, y_train)

        y_pred_before = base_model_for_comparison.predict(X_test)
        try:
            y_prob_before = base_model_for_comparison.predict_proba(X_test)[:, 1]
        except (NotImplementedError, AttributeError):
            logger.warning("Model does not have predict_proba. Post-processing may be impacted.")
            y_prob_before = y_pred_before # Use predictions as a fallback

        metrics_def = {'accuracy': accuracy_score, 'selection_rate': selection_rate, 'true_positive_rate': true_positive_rate, 'false_positive_rate': false_positive_rate}
        
        mf_before = MetricFrame(metrics=metrics_def, y_true=y_test, y_pred=y_pred_before, sensitive_features=A_test)
        plots_before = plot_metrics_to_base64(mf_before, "(Before Mitigation)")
        group_perf_before = mf_before.by_group.to_dict(orient='index')
        fairness_metrics_before = pipeline.core_analyzer.calculate_fairness_metrics(y_test, y_pred_before, A_test)

        # 4. Apply Mitigation
        logger.info("--- Mitigation Step ---")
        
        # We need a fresh model instance for in-processing/pre-processing to not affect the original
        model_for_mitigation = model if is_pretrained else RandomForestClassifier(random_state=42)
        
        y_pred_after = apply_mitigation(
            model=model_for_mitigation, 
            X_train=X_train, X_test=X_test, 
            y_train=y_train, y_test=y_test, 
            A_train=A_train, A_test=A_test,
            strategy=mitigation_strategy,
            y_prob_test_before=y_prob_before
        )
        
        # 5. Evaluate After Mitigation
        logger.info("--- Evaluation After Mitigation ---")
        mf_after = MetricFrame(metrics=metrics_def, y_true=y_test, y_pred=y_pred_after, sensitive_features=A_test)
        plots_after = plot_metrics_to_base64(mf_after, "(After Mitigation)")
        group_perf_after = mf_after.by_group.to_dict(orient='index')
        fairness_metrics_after = pipeline.core_analyzer.calculate_fairness_metrics(y_test, y_pred_after, A_test)

        # 6. Generate Reports and Recommendations
        logger.info("Generating reports...")
        bias_detected = pipeline.core_analyzer.detect_bias(fairness_metrics_before)
        recommendations = pipeline.core_analyzer.recommend_mitigation_strategies(bias_detected)

        # Calculate overall fairness scores
        overall_score_before = pipeline.core_analyzer.calculate_overall_fairness_score(fairness_metrics_before)
        overall_score_after = pipeline.core_analyzer.calculate_overall_fairness_score(fairness_metrics_after)
        
        # Estimate expected score after mitigation (for comparison)
        expected_score = pipeline.core_analyzer.estimate_post_mitigation_score(
            overall_score_before, bias_detected, mitigation_strategy
        )

        llm_data = {
            "Fairness Metrics Before Mitigation": {k: v.to_dict() for k, v in fairness_metrics_before.items()},
            "Fairness Metrics After Mitigation": {k: v.to_dict() for k, v in fairness_metrics_after.items()},
            "Group Performance Before Mitigation": group_perf_before,
            "Group Performance After Mitigation": group_perf_after,
            "Overall Fairness Score Before": overall_score_before.to_dict(),
            "Overall Fairness Score After": overall_score_after.to_dict(),
        }
        llm_report = pipeline.generate_llm_report(target_col, llm_data, mitigation_strategy)

        # 7. Structure and Return Response
        return JSONResponse(content={
            "metrics": {
                sensitive_feature: {
                    "fairness_metrics_before": {k: v.to_dict() for k, v in fairness_metrics_before.items()},
                    "fairness_metrics_after": {k: v.to_dict() for k, v in fairness_metrics_after.items()},
                    "group_performance_before": group_perf_before,
                    "group_performance_after": group_perf_after,
                    "bias_detected": bias_detected,
                    "recommendations": recommendations,
                }
            },
            "overall_fairness_scores": {
                "before_mitigation": overall_score_before.to_dict(),
                "after_mitigation": overall_score_after.to_dict(),
                "expected_after_mitigation": expected_score.to_dict(),
                "mitigation_strategy": mitigation_strategy
            },
            "plots": {
                "before_mitigation": plots_before,
                "after_mitigation": plots_after
            },
            "llm_analysis_report": llm_report
        })

    except (ValueError, RuntimeError, NotImplementedError) as e:
        logger.error(f"Validation or Runtime Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"An unexpected error occurred: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
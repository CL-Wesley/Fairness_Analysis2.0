# fairness_analyzer.py
"""
Comprehensive Fairness Analysis Backend System (V2 - Enhanced & Corrected)
Supports multiple model types, corrected fairness metrics, and advanced analysis placeholders.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple, Any, Optional
import warnings
from dataclasses import dataclass
from scipy.stats import chi2_contingency

@dataclass
class FairnessMetrics:
    """Container for fairness metrics results for a single group."""
    demographic_parity_difference: float  # Renamed for clarity
    equalized_odds_difference: float      # Corrected calculation
    equal_opportunity_difference: float   # Corrected calculation
    disparate_impact_ratio: float         # Renamed for clarity
    average_odds_difference: float
    theil_index: float

    def to_dict(self):
        return {
            'demographic_parity_difference': self.demographic_parity_difference,
            'equalized_odds_difference': self.equalized_odds_difference,
            'equal_opportunity_difference': self.equal_opportunity_difference,
            'disparate_impact_ratio': self.disparate_impact_ratio,
            'average_odds_difference': self.average_odds_difference,
            'theil_index': self.theil_index,
        }

@dataclass
class OverallFairnessScore:
    """Container for overall fairness score calculation."""
    score: float
    grade: str
    color: str
    description: str
    improvement_potential: float
    
    def to_dict(self):
        return {
            'score': self.score,
            'grade': self.grade,
            'color': self.color,
            'description': self.description,
            'improvement_potential': self.improvement_potential
        }

class FairnessAnalyzer:
    """
    Main class for fairness analysis.
    V2 includes corrected metrics, statistical significance testing, and intersectional analysis.
    """

    def __init__(self, model_type: str = 'classification'):
        self.model_type = model_type

    def _safe_score(self, value: float, multiplier: float = 100, is_ratio: bool = False) -> float:
        """
        Safely calculate score from a metric value, handling NaN and infinity cases.
        
        Args:
            value: The metric value to score
            multiplier: Scaling factor for differences (default 100 for more reasonable scaling)
            is_ratio: Whether this is a ratio metric (like disparate impact)
        
        Returns:
            Score from 0-100 where 100 is perfectly fair
        """
        if np.isnan(value) or np.isinf(value):
            return 0  # Worst possible score for undefined metrics
        
        if is_ratio:
            # For ratios like disparate impact, closer to 1 is better
            return max(0, 100 - abs(1 - value) * 100)
        else:
            # For differences, closer to 0 is better
            # Use practical significance thresholds: 20 points per 0.1 difference
            return max(0, 100 - (abs(value) / 0.1) * 20)

    def calculate_overall_fairness_score(
        self,
        metrics_by_comparison: Dict[str, FairnessMetrics]
    ) -> OverallFairnessScore:
        """
        Calculate an overall fairness score from 0-100 based on all fairness metrics.
        
        Score interpretation:
        - 90-100: Excellent (Bright Green)
        - 80-89: Good (Green)
        - 70-79: Fair (Yellow-Green)
        - 60-69: Poor (Yellow)
        - 50-59: Bad (Orange)
        - 0-49: Critical (Red)
        """
        if not metrics_by_comparison:
            return OverallFairnessScore(0, "N/A", "#808080", "No data available", 0)
        
        # Extract all metrics for scoring
        all_scores = []
        
        for comparison_key, metrics in metrics_by_comparison.items():
            # Score each metric (0-100, where 100 is perfectly fair)
            
            # Use the new safe scoring method with reasonable scaling
            dp_score = self._safe_score(metrics.demographic_parity_difference)
            eo_score = self._safe_score(metrics.equalized_odds_difference)
            eop_score = self._safe_score(metrics.equal_opportunity_difference)
            di_score = self._safe_score(metrics.disparate_impact_ratio, is_ratio=True)
            ao_score = self._safe_score(metrics.average_odds_difference)
            theil_score = self._safe_score(metrics.theil_index, multiplier=200)  # Theil index typically smaller
            
            # Weighted average of all metrics
            comparison_score = (
                dp_score * 0.25 +      # Demographic Parity - 25%
                eo_score * 0.20 +      # Equalized Odds - 20%
                eop_score * 0.15 +     # Equal Opportunity - 15%
                di_score * 0.20 +      # Disparate Impact - 20%
                ao_score * 0.10 +      # Average Odds - 10%
                theil_score * 0.10     # Theil Index - 10%
            )
            
            all_scores.append(comparison_score)
        
        # Overall score is the average across all group comparisons
        overall_score = np.mean(all_scores) if all_scores else 0
        overall_score = max(0, min(100, overall_score))  # Clamp between 0-100
        
        # Determine grade and color
        if overall_score >= 90:
            grade = "A+"
            color = "#00C851"  # Bright Green
            description = "Excellent fairness - minimal bias detected"
        elif overall_score >= 80:
            grade = "A"
            color = "#28A745"  # Green
            description = "Good fairness - low bias levels"
        elif overall_score >= 70:
            grade = "B"
            color = "#8BC34A"  # Yellow-Green
            description = "Fair fairness - moderate bias detected"
        elif overall_score >= 60:
            grade = "C"
            color = "#FFC107"  # Yellow
            description = "Poor fairness - significant bias present"
        elif overall_score >= 50:
            grade = "D"
            color = "#FF9800"  # Orange
            description = "Bad fairness - substantial bias issues"
        else:
            grade = "F"
            color = "#DC3545"  # Red
            description = "Critical fairness issues - severe bias detected"
        
        # Calculate improvement potential (how much could be gained)
        improvement_potential = 100 - overall_score
        
        return OverallFairnessScore(
            score=round(overall_score, 1),
            grade=grade,
            color=color,
            description=description,
            improvement_potential=round(improvement_potential, 1)
        )

    def estimate_post_mitigation_score(
        self,
        current_score: OverallFairnessScore,
        bias_detected: Dict[str, List[str]],
        mitigation_strategy: str
    ) -> OverallFairnessScore:
        """
        Estimate the expected fairness score after applying mitigation strategies.
        """
        # Strategy effectiveness mapping (conservative estimates)
        strategy_effectiveness = {
            'reweighing': 0.15,           # 15% improvement
            'exponentiated_gradient': 0.25, # 25% improvement
            'grid_search': 0.20,          # 20% improvement
            'postprocessing': 0.10,       # 10% improvement
            'adversarial_debiasing': 0.30, # 30% improvement
        }
        
        base_improvement = strategy_effectiveness.get(mitigation_strategy, 0.15)
        
        # Adjust based on number of bias types detected
        bias_severity = len([biases for biases in bias_detected.values() if biases])
        severity_multiplier = max(0.5, 1.0 - (bias_severity * 0.1))
        
        # Calculate expected improvement
        improvement = current_score.improvement_potential * base_improvement * severity_multiplier
        expected_score = min(100, current_score.score + improvement)
        
        # Create new score object
        if expected_score >= 90:
            grade = "A+"
            color = "#00C851"
            description = "Expected excellent fairness after mitigation"
        elif expected_score >= 80:
            grade = "A"
            color = "#28A745"
            description = "Expected good fairness after mitigation"
        elif expected_score >= 70:
            grade = "B"
            color = "#8BC34A"
            description = "Expected fair fairness after mitigation"
        elif expected_score >= 60:
            grade = "C"
            color = "#FFC107"
            description = "Expected moderate fairness after mitigation"
        elif expected_score >= 50:
            grade = "D"
            color = "#FF9800"
            description = "Expected limited fairness improvement"
        else:
            grade = "F"
            color = "#DC3545"
            description = "Significant mitigation efforts needed"
        
        return OverallFairnessScore(
            score=round(expected_score, 1),
            grade=grade,
            color=color,
            description=description,
            improvement_potential=round(100 - expected_score, 1)
        )

    def calculate_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.Series
    ) -> Dict[str, FairnessMetrics]:
        """
        Calculate all fairness metrics for given predictions and sensitive features.
        This function now calculates differences/ratios BETWEEN groups, which is the standard.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            sensitive_features: Series of sensitive feature values.

        Returns:
            A dictionary where keys are group comparisons (e.g., "groupA_vs_groupB")
            and values are FairnessMetrics objects.
        """
        unique_groups = sorted(sensitive_features.unique())
        if len(unique_groups) < 2:
            warnings.warn("Fairness metrics require at least two groups. Skipping calculation.")
            return {}

        # Calculate base rates for each group
        group_stats = {}
        for group in unique_groups:
            mask = (sensitive_features == group)
            if not np.any(mask): continue

            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            # **FIX**: Handle confusion matrix shape issues
            cm = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1])
            
            # Handle degenerate cases where confusion matrix is not 2x2
            if cm.shape != (2, 2):
                if cm.shape == (1, 1):
                    # Only one class present in this group
                    if len(np.unique(group_y_true)) == 1 and group_y_true[0] == 0:
                        # All negatives
                        tn, fp, fn, tp = cm[0,0], 0, 0, 0
                    else:
                        # All positives
                        tn, fp, fn, tp = 0, 0, 0, cm[0,0]
                else:
                    # Pad the matrix to 2x2
                    padded_cm = np.zeros((2, 2), dtype=int)
                    padded_cm[:cm.shape[0], :cm.shape[1]] = cm
                    tn, fp, fn, tp = padded_cm.ravel()
            else:
                tn, fp, fn, tp = cm.ravel()

            # **FIX**: Use NaN for undefined rates instead of 0
            total = tn + fp + fn + tp
            selection_rate = (tp + fp) / total if total > 0 else np.nan
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            
            group_stats[group] = {'sr': selection_rate, 'tpr': tpr, 'fpr': fpr, 'predictions': group_y_pred}

        # Calculate pairwise metrics
        metrics_by_comparison = {}
        # We designate the first group as the reference (unprivileged) for calculation simplicity
        # A more advanced implementation could allow user selection of the privileged group.
        ref_group = unique_groups[0]
        ref_stats = group_stats[ref_group]

        for i in range(1, len(unique_groups)):
            comp_group = unique_groups[i]
            comp_stats = group_stats[comp_group]
            
            comparison_key = f"{comp_group}_vs_{ref_group}"

            # Demographic Parity Difference (Statistical Parity)
            dpd = comp_stats['sr'] - ref_stats['sr']

            # **FIX**: Disparate Impact Ratio with proper handling of division by zero
            if not np.isnan(ref_stats['sr']) and ref_stats['sr'] > 0:
                di_ratio = comp_stats['sr'] / ref_stats['sr']
            elif not np.isnan(comp_stats['sr']) and comp_stats['sr'] > 0:
                di_ratio = np.inf  # Infinite disparity
            else:
                di_ratio = 1.0  # Both groups have 0 or NaN selection rate - assume equal

            # **FIX**: Correct Equalized Odds Difference (max of TPR and FPR difference)
            eod = max(abs(comp_stats['tpr'] - ref_stats['tpr']), abs(comp_stats['fpr'] - ref_stats['fpr']))
            
            # Equal Opportunity Difference (difference in TPR)
            eoppd = comp_stats['tpr'] - ref_stats['tpr']
            
            # Average Odds Difference
            aod = ((comp_stats['tpr'] - ref_stats['tpr']) + (comp_stats['fpr'] - ref_stats['fpr'])) / 2
            
            # **FIX**: Calculate Theil index for inequality between groups, not within all predictions
            theil = self._calculate_theil_index_between_groups(
                np.concatenate([ref_stats['predictions'], comp_stats['predictions']]),
                pd.Series([ref_group] * len(ref_stats['predictions']) + [comp_group] * len(comp_stats['predictions']))
            )

            metrics_by_comparison[comparison_key] = FairnessMetrics(
                demographic_parity_difference=dpd,
                disparate_impact_ratio=di_ratio,
                equalized_odds_difference=eod,
                equal_opportunity_difference=eoppd,
                average_odds_difference=aod,
                theil_index=theil
            )
        return metrics_by_comparison
    
    def _calculate_theil_index(self, benefits: np.ndarray) -> float:
        """
        Calculate Theil T index for a vector of benefits (e.g., predictions).
        A value of 0 is perfect equality. Higher values mean more inequality.
        
        NOTE: This is kept for backward compatibility but prefer _calculate_theil_index_between_groups
        """
        if len(benefits) == 0:
            return 0.0
        
        mean_benefit = np.mean(benefits)
        
        # Use a tolerance check for floating point numbers
        if np.isclose(mean_benefit, 0):
            return 0.0

        # Use np.where to handle log(0) case safely
        ratio = benefits / mean_benefit
        log_ratio = np.where(ratio > 0, np.log(ratio), 0)
        
        theil_t = np.mean(ratio * log_ratio)
        return float(theil_t)

    def _calculate_theil_index_between_groups(self, y_pred: np.ndarray, sensitive_features: pd.Series) -> float:
        """
        Calculate Theil index measuring inequality between groups.
        This is the proper way to measure fairness using Theil index.
        """
        groups = sensitive_features.unique()
        if len(groups) < 2:
            return 0.0
        
        # Calculate mean prediction per group
        group_means = []
        group_sizes = []
        
        for group in groups:
            mask = (sensitive_features == group)
            if np.any(mask):
                group_mean = np.mean(y_pred[mask])
                group_size = np.sum(mask)
                group_means.append(group_mean)
                group_sizes.append(group_size)
        
        if not group_means:
            return 0.0
        
        # Calculate overall mean
        overall_mean = np.average(group_means, weights=group_sizes)
        
        if np.isclose(overall_mean, 0):
            return 0.0
        
        # Calculate Theil T index between groups
        theil = 0.0
        total_size = sum(group_sizes)
        
        for mean, size in zip(group_means, group_sizes):
            if mean > 0:
                weight = size / total_size
                ratio = mean / overall_mean
                theil += weight * ratio * np.log(ratio)
        
        return float(theil)

    def detect_bias(
        self,
        metrics_by_comparison: Dict[str, FairnessMetrics],
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[str]]:
        """
        Detect bias based on fairness metrics and standard thresholds.

        Returns:
            Dictionary mapping metric names to groups/comparisons that violate thresholds.
        """
        if thresholds is None:
            thresholds = {
                'demographic_parity_difference': 0.1,
                'equalized_odds_difference': 0.1,
                'equal_opportunity_difference': 0.1,
                'disparate_impact_ratio': (0.8, 1.25),  # Use a tuple for range
                'average_odds_difference': 0.1
            }

        bias_detected = {metric: [] for metric in thresholds.keys()}

        for comparison, metrics in metrics_by_comparison.items():
            metrics_dict = metrics.to_dict()

            for metric_name, threshold in thresholds.items():
                value = metrics_dict.get(metric_name, 0.0)

                if metric_name == 'disparate_impact_ratio':
                    lower_bound, upper_bound = threshold
                    if not (lower_bound <= value <= upper_bound):
                        bias_detected[metric_name].append(comparison)
                else:
                    if abs(value) > threshold:
                        bias_detected[metric_name].append(comparison)
        return bias_detected

    def recommend_mitigation_strategies(self, bias_detected: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Generate mitigation strategy recommendations based on detected biases.
        """
        recommendations = []
        # Use a set to avoid duplicate recommendations
        added_metrics = set()

        if any(bias_detected.get('demographic_parity_difference', [])) and 'dp' not in added_metrics:
            recommendations.append({
                'metric': 'Demographic Parity',
                'issue': 'Disproportionate selection rates across groups.',
                'strategies': ['Reweighing (Pre-processing)', 'Exponentiated Gradient (In-processing)', 'Grid Search (In-processing)', 'Threshold Optimization (Post-processing)']
            })
            added_metrics.add('dp')

        if any(bias_detected.get('equalized_odds_difference', [])) and 'eo' not in added_metrics:
            recommendations.append({
                'metric': 'Equalized Odds',
                'issue': 'Disparities in true positive and false positive rates.',
                'strategies': ['Calibrated Equalized Odds (Post-processing)', 'Adversarial Debiasing (In-processing)', 'Fairness Constraints in Loss Function']
            })
            added_metrics.add('eo')
        
        if any(bias_detected.get('disparate_impact_ratio', [])) and 'di' not in added_metrics:
            recommendations.append({
                'metric': 'Disparate Impact',
                'issue': 'Adverse impact on a protected group based on selection rates.',
                'strategies': ['Disparate Impact Remover (Pre-processing)', 'Reweighing (Pre-processing)', 'Review features for proxies']
            })
            added_metrics.add('di')

        return recommendations
    
    # --- Placeholders for Advanced Features ---
    def detect_bias_with_significance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.Series,
        alpha: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Placeholder for bias detection with statistical significance (e.g., Chi-squared test).
        """
        results = {}
        unique_groups = sensitive_features.unique()
        for group in unique_groups:
            mask = (sensitive_features == group)
            contingency_table = pd.crosstab(y_true[mask], y_pred[mask])
            if contingency_table.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(contingency_table)
                results[group] = {
                    'chi2_statistic': chi2,
                    'p_value': p,
                    'is_significant': p < alpha
                }
        return {
            "statistical_significance_note": "This is a basic chi-squared test per group. A full implementation would compare groups.",
            "results": results
        }

    def analyze_intersectional_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Placeholder for analyzing bias across intersections of multiple sensitive attributes.
        Example: Race AND Gender.
        """
        if sensitive_features_df.shape[1] < 2:
            return {"error": "Intersectional analysis requires at least two sensitive feature columns."}
        
        # Create a new interaction feature
        intersectional_group = sensitive_features_df.apply(
            lambda row: '_'.join(row.values.astype(str)), axis=1
        )
        
        # Re-run the main analysis on this new intersectional group
        intersectional_metrics = self.calculate_fairness_metrics(y_true, y_pred, intersectional_group)
        
        return {
            "intersectional_analysis_note": "Analysis performed on combined sensitive attributes.",
            "metrics": {k: v.to_dict() for k,v in intersectional_metrics.items()}
        }
"""
Fairness Metrics Module
AI Ethics Assignment - Custom fairness calculations for COMPAS audit

Functions for calculating standard fairness metrics used in AI bias evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import json

# ============================================================================
# BASIC FAIRNESS METRICS
# ============================================================================

def calculate_fpr(y_true, y_pred):
    """
    Calculate False Positive Rate
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
    
    Returns:
        float: False positive rate (0-1)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr


def calculate_fnr(y_true, y_pred):
    """
    Calculate False Negative Rate
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: False negative rate (0-1)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return fnr


def calculate_tpr(y_true, y_pred):
    """
    Calculate True Positive Rate (Sensitivity/Recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: True positive rate (0-1)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return tpr


def calculate_accuracy(y_true, y_pred):
    """
    Calculate Overall Accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Accuracy (0-1)
    """
    return np.mean(y_true == y_pred)


def get_confusion_matrix_elements(y_true, y_pred):
    """
    Get all confusion matrix elements
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        dict: {'tn': int, 'fp': int, 'fn': int, 'tp': int}
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

# ============================================================================
# DISPARATE IMPACT METRICS
# ============================================================================

def disparate_impact_ratio(rate_group_a, rate_group_b):
    """
    Calculate Disparate Impact Ratio
    
    Standard: ratio >= 0.8 is considered acceptable (US EEOC 80% rule)
    Ratio < 0.8 suggests discrimination against group A
    Ratio > 1.25 suggests reverse discrimination
    
    Args:
        rate_group_a: Selection rate for group A (float 0-1)
        rate_group_b: Selection rate for group B (float 0-1)
    
    Returns:
        float: Disparate impact ratio
    """
    if rate_group_b == 0:
        return float('inf')
    return rate_group_a / rate_group_b


def calculate_selection_rates(df, prediction_col, group_col):
    """
    Calculate selection rates by group
    
    Selection rate = proportion of group predicted positive
    
    Args:
        df: DataFrame with predictions
        prediction_col: Column name of predictions (0/1)
        group_col: Column name of groups
    
    Returns:
        dict: {'group_name': selection_rate, ...}
    """
    rates = {}
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        positive_rate = group_df[prediction_col].mean()
        rates[str(group)] = float(positive_rate)
    return rates


# ============================================================================
# EQUALIZED ODDS METRICS
# ============================================================================

def equal_opportunity_difference(fnr_a, fnr_b):
    """
    Calculate Equal Opportunity Gap (based on False Negative Rate)
    
    Equalized odds requires:
    - Equal FPR across groups (False Positive Rate Parity)
    - Equal FNR across groups (True Positive Rate Parity)
    
    Target: difference < 0.05 (5%)
    
    Args:
        fnr_a: False negative rate for group A
        fnr_b: False negative rate for group B
    
    Returns:
        float: Absolute difference (0-1)
    """
    return abs(fnr_a - fnr_b)


def equal_opportunity_fpr_difference(fpr_a, fpr_b):
    """
    Calculate False Positive Rate Parity Gap
    
    Equalized Odds requirement: FPR should be similar across groups
    
    Args:
        fpr_a: False positive rate for group A
        fpr_b: False positive rate for group B
    
    Returns:
        float: Absolute difference (0-1)
    """
    return abs(fpr_a - fpr_b)


def calculate_equalized_odds_gap(y_true_a, y_pred_a, y_true_b, y_pred_b):
    """
    Calculate both FPR and FNR parity gaps
    
    Args:
        y_true_a: True labels for group A
        y_pred_a: Predicted labels for group A
        y_true_b: True labels for group B
        y_pred_b: Predicted labels for group B
    
    Returns:
        dict: {
            'fpr_gap': float,
            'fnr_gap': float,
            'fpr_a': float,
            'fnr_a': float,
            'fpr_b': float,
            'fnr_b': float
        }
    """
    fpr_a = calculate_fpr(y_true_a, y_pred_a)
    fnr_a = calculate_fnr(y_true_a, y_pred_a)
    fpr_b = calculate_fpr(y_true_b, y_pred_b)
    fnr_b = calculate_fnr(y_true_b, y_pred_b)
    
    return {
        'fpr_gap': equal_opportunity_fpr_difference(fpr_a, fpr_b),
        'fnr_gap': equal_opportunity_difference(fnr_a, fnr_b),
        'fpr_a': fpr_a,
        'fnr_a': fnr_a,
        'fpr_b': fpr_b,
        'fnr_b': fnr_b
    }

# ============================================================================
# DEMOGRAPHIC PARITY
# ============================================================================

def demographic_parity_difference(selection_rate_a, selection_rate_b):
    """
    Calculate Demographic Parity Gap
    
    Demographic Parity: P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
    i.e., selection rates should be equal across groups
    
    Target: difference < 0.05 (5%)
    
    Args:
        selection_rate_a: Proportion positive in group A
        selection_rate_b: Proportion positive in group B
    
    Returns:
        float: Absolute difference (0-1)
    """
    return abs(selection_rate_a - selection_rate_b)


# ============================================================================
# CALIBRATION METRICS
# ============================================================================

def calibration_error(y_true, y_pred_prob, group_a_mask, group_b_mask):
    """
    Calculate calibration error across groups
    
    Calibration: When model predicts X% probability,
    that outcome should occur X% of the time, for all groups
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities (0-1)
        group_a_mask: Boolean mask for group A
        group_b_mask: Boolean mask for group B
    
    Returns:
        dict: Calibration errors for each group
    """
    # Bin predictions into groups
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    calibration_a = []
    calibration_b = []
    
    for i in range(len(bins)-1):
        # Group A calibration
        in_bin_a = (y_pred_prob[group_a_mask] >= bins[i]) & (y_pred_prob[group_a_mask] < bins[i+1])
        if in_bin_a.sum() > 0:
            bin_pred_prob_a = y_pred_prob[group_a_mask][in_bin_a].mean()
            bin_actual_a = y_true[group_a_mask][in_bin_a].mean()
            calibration_a.append(abs(bin_pred_prob_a - bin_actual_a))
        
        # Group B calibration
        in_bin_b = (y_pred_prob[group_b_mask] >= bins[i]) & (y_pred_prob[group_b_mask] < bins[i+1])
        if in_bin_b.sum() > 0:
            bin_pred_prob_b = y_pred_prob[group_b_mask][in_bin_b].mean()
            bin_actual_b = y_true[group_b_mask][in_bin_b].mean()
            calibration_b.append(abs(bin_pred_prob_b - bin_actual_b))
    
    return {
        'group_a_error': np.mean(calibration_a) if calibration_a else 0,
        'group_b_error': np.mean(calibration_b) if calibration_b else 0
    }

# ============================================================================
# THEIL INDEX (Inequality Measure)
# ============================================================================

def theil_index(predictions, protected_attribute):
    """
    Calculate Theil Index - measures inequality in predictions across groups
    
    Range: 0 (perfect equality) to ln(2) (maximum inequality)
    
    Args:
        predictions: Array of predictions/scores
        protected_attribute: Array of group labels
    
    Returns:
        float: Theil index value
    """
    groups = np.unique(protected_attribute)
    theil = 0
    
    for group in groups:
        group_preds = predictions[protected_attribute == group]
        group_mean = group_preds.mean()
        overall_mean = predictions.mean()
        
        if group_mean > 0:
            theil += len(group_preds) / len(predictions) * np.log(group_mean / overall_mean)
    
    return abs(theil)

# ============================================================================
# COMPREHENSIVE BIAS REPORT
# ============================================================================

def generate_fairness_report(df, true_col, pred_col, group_col):
    """
    Generate comprehensive fairness report
    
    Args:
        df: DataFrame with predictions and groups
        true_col: Column name for true labels
        pred_col: Column name for predictions
        group_col: Column name for group membership
    
    Returns:
        dict: Comprehensive fairness metrics
    """
    report = {}
    groups = df[group_col].unique()
    
    # Overall metrics
    report['overall_accuracy'] = calculate_accuracy(df[true_col], df[pred_col])
    
    # Per-group metrics
    group_metrics = {}
    for group in groups:
        group_df = df[df[group_col] == group]
        y_true = group_df[true_col]
        y_pred = group_df[pred_col]
        
        group_metrics[str(group)] = {
            'n_samples': len(group_df),
            'accuracy': calculate_accuracy(y_true, y_pred),
            'fpr': calculate_fpr(y_true, y_pred),
            'fnr': calculate_fnr(y_true, y_pred),
            'tpr': calculate_tpr(y_true, y_pred),
            'selection_rate': y_pred.mean(),
            'confusion_matrix': get_confusion_matrix_elements(y_true, y_pred)
        }
    
    report['per_group_metrics'] = group_metrics
    
    # Disparate impact (between first two groups)
    if len(groups) >= 2:
        group_list = sorted(groups)
        rate_a = group_metrics[str(group_list[0])]['selection_rate']
        rate_b = group_metrics[str(group_list[1])]['selection_rate']
        report['disparate_impact_ratio'] = disparate_impact_ratio(rate_a, rate_b)
    
    return report


def save_report_json(report, filepath):
    """
    Save fairness report to JSON
    
    Args:
        report: Dictionary of metrics
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✓ Report saved to {filepath}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_fairness_report(report):
    """
    Pretty-print fairness report
    
    Args:
        report: Dictionary of metrics
    """
    print("\n" + "="*70)
    print("FAIRNESS METRICS REPORT")
    print("="*70)
    
    print(f"\nOverall Accuracy: {report['overall_accuracy']:.4f}")
    
    print("\nPer-Group Metrics:")
    for group, metrics in report['per_group_metrics'].items():
        print(f"\n  Group: {group}")
        print(f"    Samples: {metrics['n_samples']}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    FPR: {metrics['fpr']:.4f} (False Positive Rate)")
        print(f"    FNR: {metrics['fnr']:.4f} (False Negative Rate)")
        print(f"    TPR: {metrics['tpr']:.4f} (True Positive Rate)")
        print(f"    Selection Rate: {metrics['selection_rate']:.4f}")
    
    if 'disparate_impact_ratio' in report:
        ratio = report['disparate_impact_ratio']
        status = "✓ PASS" if 0.8 <= ratio <= 1.25 else "⚠ WARN"
        print(f"\nDisparate Impact Ratio: {ratio:.4f} {status}")
    
    print("\n" + "="*70 + "\n")
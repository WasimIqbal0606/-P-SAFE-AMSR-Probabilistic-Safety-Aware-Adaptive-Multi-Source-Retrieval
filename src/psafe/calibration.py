"""
B-P-SAFE-AMSR — Calibration Diagnostics & Reliability Analysis
Publication-grade calibration evaluation for P_gain and P_harm.

Computes:
  1. Brier Score
  2. Expected Calibration Error (ECE - standard uniform bins)
  3. Adaptive-bin ECE (equal-frequency quantile bins)
  4. AUROC (Area under ROC Curve)
  5. AUPRC (Area under Precision-Recall Curve - essential for rare harm events)
  6. Calibration Slope & Intercept (logistic calibration regression)
  7. Reliability diagram data (predicted probability vs empirical event rate)
  8. Comparison between uncalibrated, sigmoid (Platt), and isotonic scaling
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """
    Compute Expected Calibration Error (ECE) with uniform-width bins.
    Returns (ece_value, bin_details).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    n = len(y_true)
    
    if n == 0:
        return 0.0, []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            bin_weight = bin_count / n
            ece += bin_weight * abs(bin_acc - bin_conf)
            
            bin_details.append({
                "bin_idx": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_center": float((bin_lower + bin_upper) / 2.0),
                "count": bin_count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "calibration_gap": float(bin_acc - bin_conf),
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_center": float((bin_lower + bin_upper) / 2.0),
                "count": 0,
                "confidence": float((bin_lower + bin_upper) / 2.0),
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })
            
    return float(ece), bin_details


def compute_adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    """
    Compute Adaptive ECE with equal-frequency (quantile) bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
        
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_prob, quantiles)
    # Ensure strictly monotonic edges
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= low) & (y_prob <= high)
        else:
            in_bin = (y_prob >= low) & (y_prob < high)
            
        count = int(np.sum(in_bin))
        if count > 0:
            acc = float(np.mean(y_true[in_bin]))
            conf = float(np.mean(y_prob[in_bin]))
            ece += (count / n) * abs(acc - conf)
            
    return float(ece)


def compute_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Fit logistic calibration line: logit(p_true) = alpha + beta * logit(p_pred).
    Ideal slope = 1.0, ideal intercept = 0.0.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-4, 1.0 - 1e-4)
    
    if len(np.unique(y_true)) < 2:
        return 1.0, 0.0
        
    logits = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
        lr.fit(logits, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        return slope, intercept
    except Exception:
        # Fallback with light regularization
        try:
            lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)
            lr.fit(logits, y_true)
            return float(lr.coef_[0][0]), float(lr.intercept_[0])
        except Exception:
            return 1.0, 0.0


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, label_name: str = "probability") -> Dict:
    """
    Full calibration diagnostic report for a predicted probability vector.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    
    n_samples = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    pos_rate = float(n_pos / n_samples) if n_samples > 0 else 0.0
    
    # Brier Score
    brier = float(brier_score_loss(y_true, y_prob)) if n_samples > 0 else 0.0
    
    # AUROC & AUPRC
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = 0.50
        auprc = pos_rate
        
    # ECE and adaptive ECE
    ece, bin_details = compute_ece(y_true, y_prob, n_bins=10)
    adaptive_ece = compute_adaptive_ece(y_true, y_prob, n_bins=5)
    
    # Slope and Intercept
    slope, intercept = compute_calibration_slope_intercept(y_true, y_prob)
    
    return {
        "label": label_name,
        "n_samples": n_samples,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": pos_rate,
        "prevalence": pos_rate,
        "brier_score": brier,
        "ece": ece,
        "adaptive_ece": adaptive_ece,
        "auroc": auroc,
        "auprc": auprc,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_bins": bin_details,
    }


def compare_calibration_methods(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    target_name: str = "P_gain"
) -> Dict:
    """
    Train and compare Uncalibrated Logistic, Sigmoid (Platt) Calibrated, and Isotonic Calibrated models.
    Trained strictly without using test labels. Evaluates on test set.
    """
    results = {}
    
    # 1. Uncalibrated Logistic Regression
    clf_base = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf_base.fit(X_train, y_train)
    p_uncal_val = clf_base.predict_proba(X_val)[:, 1] if len(clf_base.classes_) > 1 else np.full(len(X_val), 0.5)
    p_uncal_test = clf_base.predict_proba(X_test)[:, 1] if len(clf_base.classes_) > 1 else np.full(len(X_test), 0.5)
    
    results["uncalibrated"] = {
        "val": evaluate_calibration(y_val, p_uncal_val, f"{target_name}_uncalibrated_val"),
        "test": evaluate_calibration(y_test, p_uncal_test, f"{target_name}_uncalibrated_test"),
    }
    
    # 2. Sigmoid (Platt) Calibration
    min_count = min(np.sum(y_train == 1), np.sum(y_train == 0))
    cv_folds = max(2, min(3, min_count)) if min_count >= 2 else 2
    
    try:
        clf_sigmoid = CalibratedClassifierCV(clf_base, cv=cv_folds, method="sigmoid")
        clf_sigmoid.fit(X_train, y_train)
        p_sig_val = clf_sigmoid.predict_proba(X_val)[:, 1]
        p_sig_test = clf_sigmoid.predict_proba(X_test)[:, 1]
    except Exception:
        p_sig_val = p_uncal_val
        p_sig_test = p_uncal_test
        
    results["sigmoid_platt"] = {
        "val": evaluate_calibration(y_val, p_sig_val, f"{target_name}_sigmoid_val"),
        "test": evaluate_calibration(y_test, p_sig_test, f"{target_name}_sigmoid_test"),
    }
    
    # 3. Isotonic Calibration
    try:
        clf_isotonic = CalibratedClassifierCV(clf_base, cv=cv_folds, method="isotonic")
        clf_isotonic.fit(X_train, y_train)
        p_iso_val = clf_isotonic.predict_proba(X_val)[:, 1]
        p_iso_test = clf_isotonic.predict_proba(X_test)[:, 1]
    except Exception:
        p_iso_val = p_uncal_val
        p_iso_test = p_uncal_test
        
    results["isotonic"] = {
        "val": evaluate_calibration(y_val, p_iso_val, f"{target_name}_isotonic_val"),
        "test": evaluate_calibration(y_test, p_iso_test, f"{target_name}_isotonic_test"),
    }
    
    return results

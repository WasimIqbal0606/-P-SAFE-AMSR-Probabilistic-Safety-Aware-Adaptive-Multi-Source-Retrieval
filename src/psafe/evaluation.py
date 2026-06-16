"""
B-P-SAFE-AMSR — Evaluation Manager
Sensitivity analysis (easy/hard splits) and leakage-safety checking.
Relevance threshold is configurable at dataset level.
"""
import numpy as np
from typing import Dict, List


def calculate_sensitivity_splits(dense_ndcg_scores):
    """
    Returns boolean arrays for 'easy' queries under definitions A, B, and C.
    """
    scores = np.array(dense_ndcg_scores)

    # Definition A: dense_ndcg > 0.5
    def_a_easy = scores > 0.5

    # Definition B: above dataset median
    median_score = np.median(scores)
    def_b_easy = scores > median_score

    # Definition C: top 60% easy, bottom 40% hard
    p40 = np.percentile(scores, 40)
    def_c_easy = scores > p40

    return {
        "Def_A": def_a_easy,
        "Def_B": def_b_easy,
        "Def_C": def_c_easy
    }


def analyze_sensitivity(dense_scores, psafe_scores, hybrid_scores):
    """
    Report SafeGain, EasyDeg, HardGain under all definitions.
    """
    splits = calculate_sensitivity_splits(dense_scores)
    results = {}

    for def_name, is_easy in splits.items():
        is_hard = ~is_easy

        easy_dense = dense_scores[is_easy]
        easy_psafe = psafe_scores[is_easy]

        hard_dense = dense_scores[is_hard]
        hard_psafe = psafe_scores[is_hard]

        easy_deg_psafe = max(0, np.mean(easy_dense) - np.mean(easy_psafe)) if len(easy_dense) > 0 else 0
        hard_gain_psafe = max(0, np.mean(hard_psafe) - np.mean(hard_dense)) if len(hard_dense) > 0 else 0

        results[def_name] = {
            "easy_count": int(np.sum(is_easy)),
            "hard_count": int(np.sum(is_hard)),
            "EasyDeg": float(easy_deg_psafe),
            "HardGain": float(hard_gain_psafe),
            "SafeGain": float(hard_gain_psafe - easy_deg_psafe)
        }

    return results


class EvaluationManager:
    """
    Controls evaluation mode and relevance threshold.

    Modes:
      - "within_dataset": leakage-safe train/val/test split (default)
      - "full_dataset_descriptive": descriptive only, NOT for paper claims
    """
    def __init__(self, mode="within_dataset", relevance_threshold=1):
        self.mode = mode
        self.relevance_threshold = relevance_threshold

    def check_leakage_safety(self):
        if self.mode == "full_dataset_descriptive":
            print("WARNING: full_dataset_descriptive mode. Results are descriptive only and NOT leakage-free.")
        else:
            print(f"Running in leakage-safe mode: {self.mode}")

    def get_relevance_threshold(self, dataset_name: str = "") -> int:
        """
        Return the relevance threshold for this evaluation.
        For BEIR default datasets, threshold is 1.
        Override via config for graded relevance datasets.
        """
        return self.relevance_threshold

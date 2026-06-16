import numpy as np

from psafe.feature_extractor import FEATURE_NAMES
from psafe.baselines import DenseEntropyRouter, DenseMarginRouter


def test_dense_margin_uses_feature_name_lookup():
    idx = FEATURE_NAMES.index("dense_score_gap_1_5")
    features = np.zeros((4, len(FEATURE_NAMES)))
    features[:, idx] = [0.01, 0.02, 0.4, 0.5]
    router = DenseMarginRouter()
    router.tune_on_validation(features, np.array([0.1, 0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.1, 0.1]))
    assert router.threshold >= 0.01


def test_dense_entropy_uses_feature_name_lookup():
    idx = FEATURE_NAMES.index("dense_entropy_norm")
    features = np.zeros((4, len(FEATURE_NAMES)))
    features[:, idx] = [0.1, 0.2, 0.8, 0.9]
    router = DenseEntropyRouter()
    router.tune_on_validation(features, np.array([0.1, 0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.5, 0.5]))
    assert router.threshold <= 0.9

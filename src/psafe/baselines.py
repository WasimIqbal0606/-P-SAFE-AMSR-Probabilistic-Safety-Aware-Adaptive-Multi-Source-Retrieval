"""
B-P-SAFE-AMSR — Router Baselines
Implements all baseline routers for fair comparison:
  1. Dense-only
  2. Always-on Deep Hybrid
  3. Random router
  4. Dense-margin threshold router
  5. Dense-entropy threshold router
  6. Regression-only router
  7. Classification-only router
  8. Oracle router
"""
import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from psafe.actions import Action, ACTION_NAMES
from psafe.feature_extractor import FEATURE_NAMES

# Pre-resolve feature indices once at import time
_IDX_MARGIN = FEATURE_NAMES.index("dense_score_gap_1_5")
_IDX_ENTROPY = FEATURE_NAMES.index("dense_entropy_norm")


@dataclass
class BaselineDecision:
    query_id: str
    action: int
    router_name: str
    score: float = 0.0


class DenseOnlyRouter:
    """Always choose Dense."""
    name = "Dense-only"

    def route(self, features, query_id, **kwargs):
        return BaselineDecision(query_id=query_id, action=Action.A0_DENSE.value,
                                router_name=self.name)


class AlwaysHybridRouter:
    """Always choose Deep Hybrid."""
    name = "Always-Hybrid"

    def route(self, features, query_id, **kwargs):
        return BaselineDecision(query_id=query_id, action=Action.A6_DEEP_HYBRID.value,
                                router_name=self.name)


class RandomRouter:
    """Choose Dense or Deep Hybrid with probability p."""
    name = "Random"

    def __init__(self, p_hybrid=0.5, seed=42):
        self.p_hybrid = p_hybrid
        self.rng = np.random.RandomState(seed)

    def tune_on_validation(self, val_dense_ndcg, val_hybrid_ndcg):
        """Set p_hybrid to the validation-tuned hybrid activation rate."""
        n = len(val_dense_ndcg)
        if n == 0:
            return
        # Use fraction where hybrid is better
        better = np.sum(val_hybrid_ndcg > val_dense_ndcg + 0.001)
        self.p_hybrid = float(better / n)

    def route(self, features, query_id, **kwargs):
        action = Action.A6_DEEP_HYBRID.value if self.rng.rand() < self.p_hybrid else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=self.p_hybrid)


class DenseMarginRouter:
    """Use dense_top1_score - dense_top2_score margin. Low margin → escalate."""
    name = "Dense-margin"

    def __init__(self):
        self.threshold = 0.0
        self.scaler = StandardScaler()

    def tune_on_validation(self, val_features, val_dense_ndcg, val_hybrid_ndcg):
        """Tune threshold on validation: find threshold that maximizes utility."""
        if isinstance(val_features, np.ndarray) and val_features.ndim == 2:
            if val_features.shape[1] <= _IDX_MARGIN:
                raise ValueError(f"Feature array has {val_features.shape[1]} cols; expected > {_IDX_MARGIN} for dense_score_gap_1_5")
            margins = val_features[:, _IDX_MARGIN].astype(float)
        else:
            margins = np.array([getattr(f, 'dense_score_gap_1_5',
                                        getattr(f, 'dense_score_max', 0) - getattr(f, 'dense_score_mean', 0))
                                for f in val_features])
        delta = val_hybrid_ndcg - val_dense_ndcg
        best_score = -999
        for t in np.percentile(margins, np.arange(10, 91, 5)):
            selected = margins < t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)
                self.best_val_score = score

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            if len(features) <= _IDX_MARGIN:
                raise ValueError(f"Feature vector too short: len={len(features)}, need > {_IDX_MARGIN}")
            margin = float(features[_IDX_MARGIN])
        else:
            margin = getattr(features, 'dense_score_gap_1_5',
                             getattr(features, 'dense_score_max', 0) - getattr(features, 'dense_score_mean', 0))
        action = Action.A6_DEEP_HYBRID.value if margin < self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=float(margin))


class DenseEntropyRouter:
    """Use dense score entropy. High entropy → escalate."""
    name = "Dense-entropy"

    def __init__(self):
        self.threshold = 0.5

    def tune_on_validation(self, val_features, val_dense_ndcg, val_hybrid_ndcg):
        """Tune threshold on validation."""
        if isinstance(val_features, np.ndarray) and val_features.ndim == 2:
            if val_features.shape[1] <= _IDX_ENTROPY:
                raise ValueError(f"Feature array has {val_features.shape[1]} cols; expected > {_IDX_ENTROPY} for dense_entropy_norm")
            entropies = val_features[:, _IDX_ENTROPY].astype(float)
        else:
            entropies = np.array([getattr(f, 'dense_entropy_norm', 0.5) for f in val_features])
        best_score = -999
        for t in np.percentile(entropies, np.arange(10, 91, 5)):
            selected = entropies > t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)
                self.best_val_score = score

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            if len(features) <= _IDX_ENTROPY:
                raise ValueError(f"Feature vector too short: len={len(features)}, need > {_IDX_ENTROPY}")
            entropy = float(features[_IDX_ENTROPY])
        else:
            entropy = getattr(features, 'dense_entropy_norm', 0.5)
        action = Action.A6_DEEP_HYBRID.value if entropy > self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=float(entropy))


class RegressionOnlyRouter:
    """Train regressor to predict delta_ndcg. Hybrid if predicted_delta > threshold."""
    name = "Regression-only"

    def __init__(self):
        self.model = Ridge()
        self.scaler = StandardScaler()
        self.threshold = 0.0

    def train(self, X_train, y_delta):
        X = self.scaler.fit_transform(X_train)
        self.model.fit(X, y_delta)

    def tune_on_validation(self, X_val, val_dense_ndcg, val_hybrid_ndcg):
        X = self.scaler.transform(X_val)
        preds = self.model.predict(X)
        best_score = -999
        for t in np.percentile(preds, np.arange(10, 91, 5)):
            selected = preds > t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            X = self.scaler.transform(features.reshape(1, -1))
        else:
            X = self.scaler.transform(np.array(features).reshape(1, -1))
        pred = float(self.model.predict(X)[0])
        action = Action.A6_DEEP_HYBRID.value if pred > self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=pred)


class ClassificationOnlyRouter:
    """Train classifier to predict gain_label. Hybrid if P_gain > threshold."""
    name = "Classification-only"

    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.scaler = StandardScaler()
        self.threshold = 0.5

    def train(self, X_train, y_gain):
        X = self.scaler.fit_transform(X_train)
        unique = np.unique(y_gain)
        if len(unique) < 2:
            self._prior = float(np.mean(y_gain))
            self._fitted = False
        else:
            self.model.fit(X, y_gain)
            self._fitted = True

    def tune_on_validation(self, X_val, val_dense_ndcg, val_hybrid_ndcg):
        if not getattr(self, '_fitted', True):
            return
        X = self.scaler.transform(X_val)
        probs = self.model.predict_proba(X)[:, 1]
        best_score = -999
        for t in np.arange(0.1, 0.91, 0.05):
            selected = probs > t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            X = self.scaler.transform(features.reshape(1, -1))
        else:
            X = self.scaler.transform(np.array(features).reshape(1, -1))
        if getattr(self, '_fitted', True):
            prob = float(self.model.predict_proba(X)[0, 1])
        else:
            prob = getattr(self, '_prior', 0.5)
        action = Action.A6_DEEP_HYBRID.value if prob > self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=prob)


class OracleRouter:
    """For each test query, choose the action with better true nDCG."""
    name = "Oracle"

    def __init__(self):
        self._dense_ndcg = {}
        self._hybrid_ndcg = {}

    def set_ground_truth(self, query_ids, dense_ndcg, hybrid_ndcg):
        for qid, dn, hn in zip(query_ids, dense_ndcg, hybrid_ndcg):
            self._dense_ndcg[qid] = float(dn)
            self._hybrid_ndcg[qid] = float(hn)

    def route(self, features, query_id, **kwargs):
        dn = self._dense_ndcg.get(query_id, 0.0)
        hn = self._hybrid_ndcg.get(query_id, 0.0)
        action = Action.A6_DEEP_HYBRID.value if hn > dn else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name,
                                score=max(dn, hn))


# Registry
BASELINE_ROUTERS = {
    "Dense-only": DenseOnlyRouter,
    "Always-Hybrid": AlwaysHybridRouter,
    "Random": RandomRouter,
    "Dense-margin": DenseMarginRouter,
    "Dense-entropy": DenseEntropyRouter,
    "Regression-only": RegressionOnlyRouter,
    "Classification-only": ClassificationOnlyRouter,
    "Oracle": OracleRouter,
}

"""
B-P-SAFE-AMSR — Router Baselines
Implements all baseline routers for fair and exhaustive comparison:
  1. Dense-only (A0)
  2. Always-Hybrid (A6)
  3. Random router (validation-rate driven)
  4. Matched-Budget Random router (matched to P-SAFE activation rate, 1000 allocations)
  5. Dense-margin threshold router
  6. Dense-entropy threshold router
  7. BM25-disagreement heuristic router
  8. Cost-only policy router
  9. Regression-only router (Ridge delta prediction)
  10. Classification-only router (Logistic gain prediction)
  11. Oracle upper bound
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

from psafe.actions import Action, ACTION_NAMES
from psafe.feature_extractor import FEATURE_NAMES

# Pre-resolve feature indices once at import time
_IDX_MARGIN = FEATURE_NAMES.index("dense_score_gap_1_5")
_IDX_ENTROPY = FEATURE_NAMES.index("dense_entropy_norm")
_IDX_JACCARD = FEATURE_NAMES.index("bm25_dense_overlap_jaccard_10")
_IDX_SPECIFICITY = FEATURE_NAMES.index("lexical_specificity_score")
_IDX_QLEN = FEATURE_NAMES.index("query_length_tokens")


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
                                router_name=self.name, score=0.0)


class AlwaysHybridRouter:
    """Always choose Deep Hybrid."""
    name = "Always-Hybrid"

    def route(self, features, query_id, **kwargs):
        return BaselineDecision(query_id=query_id, action=Action.A6_DEEP_HYBRID.value,
                                router_name=self.name, score=1.0)


class RandomRouter:
    """Choose Dense or Deep Hybrid with validation-tuned probability."""
    name = "Random"

    def __init__(self, p_hybrid=0.5, seed=42):
        self.p_hybrid = p_hybrid
        self.rng = np.random.RandomState(seed)

    def tune_on_validation(self, val_dense_ndcg, val_hybrid_ndcg):
        """Set p_hybrid to the validation-tuned hybrid advantage rate."""
        n = len(val_dense_ndcg)
        if n == 0:
            return
        better = np.sum(val_hybrid_ndcg > val_dense_ndcg + 0.001)
        self.p_hybrid = float(better / n)

    def route(self, features, query_id, **kwargs):
        action = Action.A6_DEEP_HYBRID.value if self.rng.rand() < self.p_hybrid else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=self.p_hybrid)


class MatchedBudgetRandomRouter:
    """
    Matched-Activation / Matched-Budget Random Baseline.
    Chooses exactly the same fraction / count k of test queries to escalate as P-SAFE,
    repeated over exactly 1000 independent random allocations for paper evidence.
    """
    name = "Matched-Budget-Random"

    def __init__(self, target_activation_rate: float = 0.5, seed: int = 42):
        self.target_activation_rate = float(target_activation_rate)
        self.seed = seed

    def evaluate_batch(
        self,
        query_ids: List[str],
        dense_ndcg: np.ndarray,
        hybrid_ndcg: np.ndarray,
        target_k: Optional[int] = None,
        seed: int = 42
    ) -> Dict:
        """
        Evaluate a single random allocation choosing exactly target_k queries.
        """
        n = len(query_ids)
        if target_k is None:
            target_k = int(np.round(self.target_activation_rate * n))
        target_k = max(0, min(n, target_k))

        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        escalated_indices = set(perm[:target_k])

        ndcg_scores = np.zeros(n)
        actions = []
        for i in range(n):
            if i in escalated_indices:
                ndcg_scores[i] = hybrid_ndcg[i]
                actions.append(Action.A6_DEEP_HYBRID.value)
            else:
                ndcg_scores[i] = dense_ndcg[i]
                actions.append(Action.A0_DENSE.value)

        mean_ndcg = float(np.mean(ndcg_scores))
        act_rate = float(target_k / n) if n > 0 else 0.0
        return {
            "mean_ndcg": mean_ndcg,
            "ndcg_scores": ndcg_scores,
            "actions": actions,
            "hybrid_activation": act_rate,
            "k": target_k,
            "seed": seed
        }

    def evaluate_multi_seed(
        self,
        query_ids: List[str],
        dense_ndcg: np.ndarray,
        hybrid_ndcg: np.ndarray,
        target_k: Optional[int] = None,
        n_repetitions: int = 1000,
        seed_base: int = 42
    ) -> Dict:
        """
        Run across n_repetitions (1000 for paper evidence) to obtain distribution metrics.
        """
        n = len(query_ids)
        if target_k is None:
            target_k = int(np.round(self.target_activation_rate * n))
        target_k = max(0, min(n, target_k))

        means = []
        for rep in range(n_repetitions):
            rep_seed = seed_base + rep * 1000 + 7
            res = self.evaluate_batch(query_ids, dense_ndcg, hybrid_ndcg, target_k=target_k, seed=rep_seed)
            means.append(res["mean_ndcg"])

        means_arr = np.array(means)
        mean_val = float(np.mean(means_arr))
        std_val = float(np.std(means_arr, ddof=1))
        ci_lower = float(np.percentile(means_arr, 2.5))
        ci_upper = float(np.percentile(means_arr, 97.5))
        p5 = float(np.percentile(means_arr, 5.0))
        p95 = float(np.percentile(means_arr, 95.0))

        return {
            "router_name": self.name,
            "target_activation_rate": float(target_k / n) if n > 0 else 0.0,
            "target_k": target_k,
            "n_queries": n,
            "n_repetitions": n_repetitions,
            "mean_ndcg": mean_val,
            "std_ndcg": std_val,
            "ci_95": [ci_lower, ci_upper],
            "p5_p95": [p5, p95],
            "min_ndcg": float(np.min(means_arr)),
            "max_ndcg": float(np.max(means_arr)),
        }


class DenseMarginRouter:
    """Use dense_top1_score - dense_top2_score margin. Low margin → escalate."""
    name = "Dense-margin"

    def __init__(self):
        self.threshold = 0.0

    def tune_on_validation(self, val_features, val_dense_ndcg, val_hybrid_ndcg):
        """Tune threshold on validation: find threshold that maximizes validation nDCG."""
        if isinstance(val_features, np.ndarray) and val_features.ndim == 2:
            margins = val_features[:, _IDX_MARGIN].astype(float)
        else:
            margins = np.array([getattr(f, 'dense_score_gap_1_5', 0.0) for f in val_features])
            
        best_score = -999.0
        self.threshold = float(np.median(margins))
        for t in np.percentile(margins, np.arange(5, 96, 5)):
            selected = margins < t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)
                self.best_val_score = score

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            margin = float(features[_IDX_MARGIN])
        else:
            margin = float(getattr(features, 'dense_score_gap_1_5', 0.0))
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
            entropies = val_features[:, _IDX_ENTROPY].astype(float)
        else:
            entropies = np.array([getattr(f, 'dense_entropy_norm', 0.5) for f in val_features])
        best_score = -999.0
        self.threshold = float(np.median(entropies))
        for t in np.percentile(entropies, np.arange(5, 96, 5)):
            selected = entropies > t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)
                self.best_val_score = score

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            entropy = float(features[_IDX_ENTROPY])
        else:
            entropy = float(getattr(features, 'dense_entropy_norm', 0.5))
        action = Action.A6_DEEP_HYBRID.value if entropy > self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=float(entropy))


class BM25DisagreementRouter:
    """
    BM25 <-> Dense Disagreement Heuristic Router.
    Escalates to Hybrid when Dense and BM25 retrieval lists disagree strongly
    (low Jaccard overlap / high novelty).
    """
    name = "BM25-disagreement"

    def __init__(self):
        self.threshold = 0.3

    def tune_on_validation(self, val_features, val_dense_ndcg, val_hybrid_ndcg):
        if isinstance(val_features, np.ndarray) and val_features.ndim == 2:
            jaccards = val_features[:, _IDX_JACCARD].astype(float)
        else:
            jaccards = np.array([getattr(f, 'bm25_dense_jaccard_10', 0.3) for f in val_features])
        best_score = -999.0
        self.threshold = float(np.median(jaccards))
        for t in np.percentile(jaccards, np.arange(5, 96, 5)):
            # Disagreement means low overlap: escalate when jaccard < threshold
            selected = jaccards < t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            jaccard = float(features[_IDX_JACCARD])
        else:
            jaccard = float(getattr(features, 'bm25_dense_jaccard_10', 0.3))
        # Escalate when overlap is low (strong disagreement)
        action = Action.A6_DEEP_HYBRID.value if jaccard < self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=float(1.0 - jaccard))


class CostOnlyRouter:
    """
    Cost-Only Policy Router.
    Escalates only for queries where lexical specificity / length indicates manageable reranking cost.
    """
    name = "Cost-only"

    def __init__(self):
        self.threshold = 0.5

    def tune_on_validation(self, val_features, val_dense_ndcg, val_hybrid_ndcg):
        if isinstance(val_features, np.ndarray) and val_features.ndim == 2:
            scores = val_features[:, _IDX_SPECIFICITY].astype(float)
        else:
            scores = np.array([getattr(f, 'query_lexical_specificity', 0.5) for f in val_features])
        best_score = -999.0
        self.threshold = float(np.median(scores))
        for t in np.percentile(scores, np.arange(5, 96, 5)):
            selected = scores > t
            ndcg = np.where(selected, val_hybrid_ndcg, val_dense_ndcg)
            score = float(np.mean(ndcg))
            if score > best_score:
                best_score = score
                self.threshold = float(t)

    def route(self, features, query_id, **kwargs):
        if isinstance(features, np.ndarray):
            val = float(features[_IDX_SPECIFICITY])
        else:
            val = float(getattr(features, 'query_lexical_specificity', 0.5))
        action = Action.A6_DEEP_HYBRID.value if val > self.threshold else Action.A0_DENSE.value
        return BaselineDecision(query_id=query_id, action=action,
                                router_name=self.name, score=float(val))


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
        best_score = -999.0
        self.threshold = 0.0
        for t in np.percentile(preds, np.arange(5, 96, 5)):
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
        self._fitted = False
        self._prior = 0.5

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
        if not self._fitted:
            return
        X = self.scaler.transform(X_val)
        probs = self.model.predict_proba(X)[:, 1]
        best_score = -999.0
        self.threshold = 0.5
        for t in np.arange(0.05, 0.96, 0.05):
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
        if self._fitted:
            prob = float(self.model.predict_proba(X)[0, 1])
        else:
            prob = self._prior
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
    "Matched-Budget-Random": MatchedBudgetRandomRouter,
    "Dense-margin": DenseMarginRouter,
    "Dense-entropy": DenseEntropyRouter,
    "BM25-disagreement": BM25DisagreementRouter,
    "Cost-only": CostOnlyRouter,
    "Regression-only": RegressionOnlyRouter,
    "Classification-only": ClassificationOnlyRouter,
    "Oracle": OracleRouter,
}

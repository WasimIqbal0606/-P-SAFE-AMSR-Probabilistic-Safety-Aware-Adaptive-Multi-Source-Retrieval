"""
Safe-AMSR-SE v4 — Multi-Action Adaptive Router
Decides per-query retrieval strategy from 5 possible actions.

Actions:
  A0 = Dense only              (cost: ~0.05ms)
  A1 = Dense + BM25            (cost: ~5ms)
  A2 = Dense + Graph           (cost: ~10ms)
  A3 = Dense + BM25 + Graph    (cost: ~15ms)
  A4 = Full (D+BM25+Graph+CE)  (cost: ~1300ms)

Router Types:
  1. RuleBasedRouter     — threshold on dense score features
  2. LearnedRouter       — binary classifier (LR / RF / GB)
  3. MultiActionRouter   — utility-maximizing multi-class router
  4. CostAwareRouter     — explicit latency penalty
  5. OracleRouter        — cheats: picks best action (upper bound)
  6. RandomRouter        — random at same activation rate (lower bound)

Utility Objective:
  U(action, query) = E[delta_ndcg(action)] - lambda_lat * latency(action) - lambda_harm * degradation(action)
"""

import numpy as np
import json, os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

from .feature_extractor import RoutingFeatures, FEATURE_NAMES


class Action(IntEnum):
    DENSE_ONLY = 0
    DENSE_BM25 = 1
    DENSE_GRAPH = 2
    DENSE_BM25_GRAPH = 3
    FULL_HYBRID = 4


ACTION_NAMES = {
    Action.DENSE_ONLY: "Dense",
    Action.DENSE_BM25: "Dense+BM25",
    Action.DENSE_GRAPH: "Dense+Graph",
    Action.DENSE_BM25_GRAPH: "Dense+BM25+Graph",
    Action.FULL_HYBRID: "Full(D+B+G+CE)",
}

# Approximate latency for each action (ms)
ACTION_LATENCY = {
    Action.DENSE_ONLY: 0.05,
    Action.DENSE_BM25: 5.0,
    Action.DENSE_GRAPH: 10.0,
    Action.DENSE_BM25_GRAPH: 15.0,
    Action.FULL_HYBRID: 1300.0,
}


@dataclass
class RouterDecision:
    query_id: str
    action: Action
    confidence: float = 0.0
    predicted_hard_prob: float = 0.0
    action_utilities: Optional[Dict[int, float]] = None
    features_used: Optional[Dict[str, float]] = None


@dataclass
class RouterTrainingData:
    feature_matrix: np.ndarray         # (N, F)
    action_ndcg: Dict[int, np.ndarray] # {action_id: ndcg array}
    dense_ndcg: np.ndarray
    full_ndcg: np.ndarray
    labels_hard: np.ndarray            # 1 if any expansion helps
    labels_safe: np.ndarray            # 1 if Dense is fine
    labels_easy_harm: np.ndarray
    query_ids: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Rule-Based Router
# ═══════════════════════════════════════════════════════════════════════

class RuleBasedRouter:
    def __init__(self, top1_threshold=0.35, margin_threshold=0.03):
        self.top1_threshold = top1_threshold
        self.margin_threshold = margin_threshold
        self.decisions: List[RouterDecision] = []
        self.is_trained = False

    def route(self, features: RoutingFeatures) -> RouterDecision:
        f = features.features
        top1 = f.get("dense_top1_score", 0)
        margin = f.get("dense_top1_top2_margin", 0)

        if top1 > self.top1_threshold and margin > self.margin_threshold:
            action, hp = Action.DENSE_ONLY, 0.1
        elif top1 > self.top1_threshold * 0.8:
            action, hp = Action.DENSE_BM25, 0.4
        elif top1 > self.top1_threshold * 0.5:
            action, hp = Action.DENSE_BM25_GRAPH, 0.6
        else:
            action, hp = Action.FULL_HYBRID, 0.8

        d = RouterDecision(query_id=features.query_id, action=action,
                           confidence=1.0 - hp, predicted_hard_prob=hp,
                           features_used=features.features)
        self.decisions.append(d)
        return d

    def get_stats(self): return _router_stats("rule_based", self.decisions)
    def reset(self): self.decisions.clear()


# ═══════════════════════════════════════════════════════════════════════
# Learned Router (Binary: Dense vs Full)
# ═══════════════════════════════════════════════════════════════════════

class LearnedRouter:
    def __init__(self, model_type="logistic", safety_threshold=0.5, feature_names=None):
        self.model_type = model_type
        self.safety_threshold = safety_threshold
        self.feature_names = feature_names or FEATURE_NAMES
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.decisions: List[RouterDecision] = []
        self.train_stats: Dict = {}

    def train(self, training_data: RouterTrainingData, val_split=0.2):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X, y = training_data.feature_matrix, training_data.labels_hard
        if len(X) < 10:
            print("   Warning: Too few training samples"); return

        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_s, y, test_size=val_split, random_state=42,
                stratify=y if len(set(y)) > 1 else None)
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(X_s, y, test_size=val_split, random_state=42)

        self.model = self._make_model()
        self.model.fit(X_tr, y_tr)
        self.is_trained = True

        val_pred = self.model.predict(X_va)
        val_prob = self.model.predict_proba(X_va)[:, 1] if hasattr(self.model, 'predict_proba') else val_pred.astype(float)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        self.train_stats = {
            "model_type": self.model_type,
            "n_train": len(X_tr), "n_val": len(X_va), "n_features": X_tr.shape[1],
            "n_hard_train": int(np.sum(y_tr)), "n_hard_val": int(np.sum(y_va)),
            "val_accuracy": float(accuracy_score(y_va, val_pred)),
            "val_precision": float(precision_score(y_va, val_pred, zero_division=0)),
            "val_recall": float(recall_score(y_va, val_pred, zero_division=0)),
            "val_f1": float(f1_score(y_va, val_pred, zero_division=0)),
            "val_probs": val_prob.tolist(), "val_labels": y_va.tolist(),
        }

        # Feature importances
        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
            names = self.feature_names[:len(imp)]
            top = sorted(zip(names, imp), key=lambda x: -x[1])[:10]
            self.train_stats["top_features"] = [{"name": n, "importance": float(v)} for n, v in top]
        elif hasattr(self.model, 'coef_'):
            coefs = np.abs(self.model.coef_[0])
            names = self.feature_names[:len(coefs)]
            top = sorted(zip(names, coefs), key=lambda x: -x[1])[:10]
            self.train_stats["top_features"] = [{"name": n, "importance": float(v)} for n, v in top]

        print(f"   Router trained ({self.model_type}): "
              f"acc={self.train_stats['val_accuracy']:.3f}, "
              f"f1={self.train_stats['val_f1']:.3f}")

    def _make_model(self):
        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
        elif self.model_type == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    def tune_threshold(self, features, dense_ndcg, full_ndcg, lambda_harm=2.0):
        if not self.is_trained: return self.safety_threshold
        X_s = self.scaler.transform(features)
        probs = self.model.predict_proba(X_s)[:, 1]
        best_t, best_sg = 0.5, -float('inf')
        easy = dense_ndcg > 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            sim = np.where(probs >= t, full_ndcg, dense_ndcg)
            ed = np.mean(np.minimum(sim[easy] - dense_ndcg[easy], 0)) if np.sum(easy) > 0 else 0
            hg = np.mean(np.maximum(sim[~easy] - dense_ndcg[~easy], 0)) if np.sum(~easy) > 0 else 0
            sg = hg + ed  # ed is already negative
            sg = hg - lambda_harm * max(0, -ed)
            if sg > best_sg: best_sg, best_t = sg, t
        self.safety_threshold = best_t
        print(f"   Tuned threshold: {best_t:.2f} (SafeGain={best_sg:.4f})")
        return best_t

    def route(self, features: RoutingFeatures) -> RouterDecision:
        if not self.is_trained:
            return RouterDecision(query_id=features.query_id, action=Action.DENSE_ONLY, confidence=0.5)
        x = features.to_array(self.feature_names).reshape(1, -1)
        hp = float(self.model.predict_proba(self.scaler.transform(x))[0, 1])
        action = Action.FULL_HYBRID if hp >= self.safety_threshold else Action.DENSE_ONLY
        d = RouterDecision(query_id=features.query_id, action=action,
                           confidence=1.0 - abs(hp - 0.5) * 2, predicted_hard_prob=hp,
                           features_used=features.features)
        self.decisions.append(d)
        return d

    def get_stats(self):
        s = _router_stats(f"learned_{self.model_type}", self.decisions)
        s["safety_threshold"] = self.safety_threshold
        s["train_stats"] = self.train_stats
        return s

    def reset(self): self.decisions.clear()


# ═══════════════════════════════════════════════════════════════════════
# Multi-Action Utility Router (v4 novelty)
# ═══════════════════════════════════════════════════════════════════════

class MultiActionRouter:
    """
    Selects from 5 actions by maximizing expected utility:
      U(a,q) = E[delta_ndcg(a,q)] - lambda_lat * latency(a) - lambda_harm * P(degrade|a,q) * deg_magnitude
    
    Training: for each query in train set, we know nDCG for all 5 actions.
    We train a multi-class classifier where label = argmax utility action.
    """

    def __init__(self, model_type="gradient_boost", lambda_lat=0.00005,
                 lambda_harm=2.0, feature_names=None):
        self.model_type = model_type
        self.lambda_lat = lambda_lat
        self.lambda_harm = lambda_harm
        self.feature_names = feature_names or FEATURE_NAMES
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.decisions: List[RouterDecision] = []
        self.train_stats: Dict = {}

    def train(self, training_data: RouterTrainingData, val_split=0.2):
        """Train multi-action router using utility-optimal labels."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        X = training_data.feature_matrix
        dense_ndcg = training_data.dense_ndcg
        action_ndcg = training_data.action_ndcg
        n = len(X)

        # Compute utility for each action
        utilities = np.zeros((n, len(Action)))
        for a in Action:
            if a.value in action_ndcg:
                a_ndcg = action_ndcg[a.value]
                delta = a_ndcg - dense_ndcg
                lat_cost = self.lambda_lat * ACTION_LATENCY[a]
                harm = self.lambda_harm * np.maximum(-delta, 0)
                utilities[:, a.value] = delta - lat_cost - harm
            else:
                utilities[:, a.value] = -999  # unavailable action

        # Label = argmax utility
        labels = np.argmax(utilities, axis=1).astype(np.int32)
        unique_labels = np.unique(labels)
        print(f"   Multi-action labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_s, labels, test_size=val_split, random_state=42,
                stratify=labels if len(unique_labels) > 1 else None)
        except ValueError:
            X_tr, X_va, y_tr, y_va = train_test_split(X_s, labels, test_size=val_split, random_state=42)

        # Multi-class classifier
        if self.model_type == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=42)

        self.model.fit(X_tr, y_tr)
        self.is_trained = True

        val_pred = self.model.predict(X_va)
        from sklearn.metrics import accuracy_score
        self.train_stats = {
            "model_type": f"multi_action_{self.model_type}",
            "n_train": len(X_tr), "n_val": len(X_va),
            "n_actions_used": len(unique_labels),
            "val_accuracy": float(accuracy_score(y_va, val_pred)),
            "lambda_lat": self.lambda_lat, "lambda_harm": self.lambda_harm,
            "action_distribution_train": {int(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
        }
        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
            names = self.feature_names[:len(imp)]
            top = sorted(zip(names, imp), key=lambda x: -x[1])[:10]
            self.train_stats["top_features"] = [{"name": n, "importance": float(v)} for n, v in top]

        print(f"   Multi-action router trained: acc={self.train_stats['val_accuracy']:.3f}")

    def route(self, features: RoutingFeatures) -> RouterDecision:
        if not self.is_trained:
            return RouterDecision(query_id=features.query_id, action=Action.DENSE_ONLY)
        x = features.to_array(self.feature_names).reshape(1, -1)
        x_s = self.scaler.transform(x)
        pred = int(self.model.predict(x_s)[0])
        action = Action(pred) if pred in [a.value for a in Action] else Action.DENSE_ONLY

        proba = self.model.predict_proba(x_s)[0] if hasattr(self.model, 'predict_proba') else np.zeros(5)
        utils = {int(i): float(p) for i, p in enumerate(proba)}

        d = RouterDecision(query_id=features.query_id, action=action,
                           confidence=float(max(proba)) if len(proba) > 0 else 0.5,
                           predicted_hard_prob=float(1 - proba[0]) if len(proba) > 0 else 0.5,
                           action_utilities=utils, features_used=features.features)
        self.decisions.append(d)
        return d

    def get_stats(self):
        s = _router_stats("multi_action", self.decisions)
        s["train_stats"] = self.train_stats
        return s

    def reset(self): self.decisions.clear()


# ═══════════════════════════════════════════════════════════════════════
# Cost-Aware Router
# ═══════════════════════════════════════════════════════════════════════

class CostAwareRouter(LearnedRouter):
    def __init__(self, model_type="logistic", safety_threshold=0.5,
                 lambda_latency=0.0001, lambda_harm=2.0):
        super().__init__(model_type, safety_threshold)
        self.lambda_latency = lambda_latency
        self.lambda_harm = lambda_harm

    def route(self, features: RoutingFeatures) -> RouterDecision:
        if not self.is_trained:
            return RouterDecision(query_id=features.query_id, action=Action.DENSE_ONLY)
        x = features.to_array(self.feature_names).reshape(1, -1)
        hp = float(self.model.predict_proba(self.scaler.transform(x))[0, 1])
        avg_gain = self.train_stats.get("avg_hard_gain", 0.05)
        avg_harm = self.train_stats.get("avg_easy_harm", 0.03)
        net = hp * avg_gain - self.lambda_latency * ACTION_LATENCY[Action.FULL_HYBRID] - self.lambda_harm * (1 - hp) * avg_harm
        action = Action.FULL_HYBRID if net > 0 and hp >= self.safety_threshold else Action.DENSE_ONLY
        d = RouterDecision(query_id=features.query_id, action=action,
                           confidence=abs(net), predicted_hard_prob=hp, features_used=features.features)
        self.decisions.append(d)
        return d


# ═══════════════════════════════════════════════════════════════════════
# Oracle Router (Upper Bound)
# ═══════════════════════════════════════════════════════════════════════

class OracleRouter:
    def __init__(self):
        self.decisions: List[RouterDecision] = []
        self._oracle_map: Dict[str, Action] = {}
        self.is_trained = True

    def set_oracle_labels(self, query_ids, dense_ndcg, full_ndcg,
                          action_ndcg: Dict[int, np.ndarray] = None):
        """Pre-compute oracle: picks action with highest nDCG per query."""
        if action_ndcg:
            for qi, qid in enumerate(query_ids):
                best_a, best_n = Action.DENSE_ONLY, dense_ndcg[qi]
                for a_val, ndcg_arr in action_ndcg.items():
                    if ndcg_arr[qi] > best_n + 1e-8:
                        best_n = ndcg_arr[qi]
                        best_a = Action(a_val)
                self._oracle_map[qid] = best_a
        else:
            for qid, dn, fn in zip(query_ids, dense_ndcg, full_ndcg):
                self._oracle_map[qid] = Action.FULL_HYBRID if fn > dn + 1e-8 else Action.DENSE_ONLY

    def route(self, features: RoutingFeatures) -> RouterDecision:
        a = self._oracle_map.get(features.query_id, Action.DENSE_ONLY)
        d = RouterDecision(query_id=features.query_id, action=a, confidence=1.0,
                           predicted_hard_prob=1.0 if a != Action.DENSE_ONLY else 0.0)
        self.decisions.append(d)
        return d

    def get_stats(self): return _router_stats("oracle", self.decisions)
    def reset(self): self.decisions.clear()


# ═══════════════════════════════════════════════════════════════════════
# Random Router (Lower Bound)
# ═══════════════════════════════════════════════════════════════════════

class RandomRouter:
    def __init__(self, hybrid_rate=0.3, seed=42):
        self.hybrid_rate = hybrid_rate
        self.rng = np.random.default_rng(seed)
        self.decisions: List[RouterDecision] = []
        self.is_trained = True

    def route(self, features: RoutingFeatures) -> RouterDecision:
        a = Action.FULL_HYBRID if self.rng.random() < self.hybrid_rate else Action.DENSE_ONLY
        d = RouterDecision(query_id=features.query_id, action=a, confidence=0.5,
                           predicted_hard_prob=self.hybrid_rate)
        self.decisions.append(d)
        return d

    def get_stats(self):
        s = _router_stats("random", self.decisions)
        s["target_hybrid_rate"] = self.hybrid_rate
        return s

    def reset(self): self.decisions.clear()


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _router_stats(rtype: str, decisions: List[RouterDecision]) -> Dict:
    if not decisions:
        return {"router_type": rtype}
    actions = [d.action for d in decisions]
    n = len(actions)
    dist = {}
    for a in Action:
        cnt = sum(1 for x in actions if x == a)
        if cnt > 0:
            dist[ACTION_NAMES[a]] = cnt
    return {
        "router_type": rtype, "total_queries": n,
        "dense_only_rate": sum(1 for a in actions if a == Action.DENSE_ONLY) / n,
        "hybrid_rate": sum(1 for a in actions if a != Action.DENSE_ONLY) / n,
        "action_distribution": dist,
    }


def compute_safety_metrics(dense_ndcg, routed_ndcg, easy_mask, method_name=""):
    hard_mask = ~easy_mask
    deltas = routed_ndcg - dense_ndcg
    eps = 1e-8
    n_easy, n_hard = int(np.sum(easy_mask)), int(np.sum(hard_mask))
    easy_d = deltas[easy_mask] if n_easy > 0 else np.array([0.0])
    hard_d = deltas[hard_mask] if n_hard > 0 else np.array([0.0])
    easy_deg = float(-np.mean(np.minimum(easy_d, 0)))
    hard_gain = float(np.mean(np.maximum(hard_d, 0)))
    return {
        "method": method_name, "safe_gain": hard_gain - easy_deg,
        "easy_degradation_mean": easy_deg,
        "easy_degradation_rate": float(np.mean(easy_d < -eps)) if n_easy > 0 else 0.0,
        "hard_gain_mean": hard_gain,
        "hard_gain_rate": float(np.mean(hard_d > eps)) if n_hard > 0 else 0.0,
        "net_ndcg_gain": float(np.mean(deltas)),
        "dense_preservation_rate": float(np.mean(deltas >= -eps)),
        "missed_hard_query_rate": float(np.mean(np.abs(hard_d) <= eps)) if n_hard > 0 else 0,
        "n_easy": n_easy, "n_hard": n_hard,
        "pct_routed_hybrid": 0.0, "pct_preserved_dense": 0.0,
    }


def build_training_data(feature_list, dense_ndcg, full_ndcg, query_ids,
                         action_ndcg=None, success_threshold=0.5,
                         benefit_epsilon=0.01, harm_epsilon=0.01):
    n = len(feature_list)
    X = np.zeros((n, len(FEATURE_NAMES)))
    for i, f in enumerate(feature_list):
        X[i] = f.to_array(FEATURE_NAMES)
    labels_hard = (full_ndcg > dense_ndcg + benefit_epsilon).astype(np.int32)
    labels_safe = (dense_ndcg >= success_threshold).astype(np.int32)
    labels_easy_harm = ((full_ndcg < dense_ndcg - harm_epsilon) & (dense_ndcg >= success_threshold)).astype(np.int32)
    print(f"   Training labels: {np.sum(labels_hard)} hard, "
          f"{np.sum(labels_safe)} safe, {np.sum(labels_easy_harm)} easy_harm")
    return RouterTrainingData(
        feature_matrix=X, action_ndcg=action_ndcg or {0: dense_ndcg, 4: full_ndcg},
        dense_ndcg=dense_ndcg, full_ndcg=full_ndcg,
        labels_hard=labels_hard, labels_safe=labels_safe, labels_easy_harm=labels_easy_harm,
        query_ids=query_ids, feature_names=FEATURE_NAMES,
    )

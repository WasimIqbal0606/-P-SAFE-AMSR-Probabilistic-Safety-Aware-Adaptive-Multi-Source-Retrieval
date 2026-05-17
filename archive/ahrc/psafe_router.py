"""
P-SAFE-AMSR: Probabilistic Safety-Aware Adaptive Multi-Source Retrieval

FIX 3: PriorProbabilityModel prevents probability collapse on single-class data.
FIX 4: allow_forced_hybrid=False by default; forced hybrid requires explicit opt-in.
FIX 5: A0 Dense always has utility=0, p_gain=0, p_harm=0; no models trained for it.
"""
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

from .feature_extractor import RoutingFeatures, FEATURE_NAMES

class Action(IntEnum):
    A0_DENSE = 0
    A1_DENSE_BM25 = 1
    A2_DENSE_GRAPH = 2
    A3_DENSE_BM25_GRAPH = 3
    A4_DENSE_BM25_CE = 4
    A5_DENSE_BM25_GRAPH_CE = 5
    A6_DEEP_HYBRID = 6

ACTION_NAMES = {
    Action.A0_DENSE: "Dense",
    Action.A1_DENSE_BM25: "Dense+BM25",
    Action.A2_DENSE_GRAPH: "Dense+Graph",
    Action.A3_DENSE_BM25_GRAPH: "Dense+BM25+Graph",
    Action.A4_DENSE_BM25_CE: "Dense+BM25+CE",
    Action.A5_DENSE_BM25_GRAPH_CE: "Dense+BM25+Graph+CE",
    Action.A6_DEEP_HYBRID: "Deep Hybrid",
}

# Empirical approximate latency for each action (ms)
ACTION_LATENCY = {
    Action.A0_DENSE: 0.05,
    Action.A1_DENSE_BM25: 5.0,
    Action.A2_DENSE_GRAPH: 10.0,
    Action.A3_DENSE_BM25_GRAPH: 15.0,
    Action.A4_DENSE_BM25_CE: 600.0,
    Action.A5_DENSE_BM25_GRAPH_CE: 1300.0,
    Action.A6_DEEP_HYBRID: 2500.0,
}

@dataclass
class PSafeDecision:
    query_id: str
    action: Action
    expected_utility: float = 0.0
    p_gain: float = 0.0
    p_harm: float = 0.0
    lcb_utility: float = 0.0
    action_utilities: Optional[Dict[int, float]] = None
    action_p_gain: Optional[Dict[int, float]] = None
    action_p_harm: Optional[Dict[int, float]] = None
    action_pred_delta: Optional[Dict[int, float]] = None
    action_pred_lat: Optional[Dict[int, float]] = None
    features_used: Optional[Dict[str, float]] = None
    rejected_reasons: Dict[int, str] = field(default_factory=dict)
    final_decision_reason: str = ""

@dataclass
class PSafeTrainingData:
    feature_matrix: np.ndarray         
    action_ndcg: Dict[int, np.ndarray] 
    action_latency: Dict[int, np.ndarray]
    query_ids: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


# ── FIX 3: PriorProbabilityModel ────────────────────────────────────
class PriorProbabilityModel:
    """Fallback model when class counts are too small for proper training."""
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-4, 1 - 1e-4))

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([
            np.full(n, 1 - self.p),
            np.full(n, self.p)
        ])

    def predict(self, X):
        return np.full(len(X), int(self.p >= 0.5))


class PSafeRouter:
    def __init__(self, 
                 lambda_lat=0.00005, 
                 lambda_harm=2.0, 
                 epsilon_gain=0.01, 
                 epsilon_harm=0.01,
                 harm_threshold=0.3,
                 gain_threshold=0.5,
                 n_bootstrap=10,
                 use_lcb_safety=False,
                 min_hybrid_rate=0.0,
                 allow_forced_hybrid=False,  # FIX 4: default False
                 feature_names=None):
        self.lambda_lat = lambda_lat
        self.lambda_harm = lambda_harm
        self.epsilon_gain = epsilon_gain
        self.epsilon_harm = epsilon_harm
        self.harm_threshold = harm_threshold
        self.gain_threshold = gain_threshold
        self.n_bootstrap = n_bootstrap
        self.use_lcb_safety = use_lcb_safety
        self.min_hybrid_rate = min_hybrid_rate
        self.allow_forced_hybrid = allow_forced_hybrid  # FIX 4
        self.feature_names = feature_names or FEATURE_NAMES
        
        self.models = {}
        self.scaler = None
        self.is_trained = False
        self.decisions: List[PSafeDecision] = []
        self.train_stats: Dict = {}
        self.calibration_data: Dict = {}
        self.class_balance: Dict = {}  # FIX 3: diagnostic output
        self.rejected_counts = {"high_P_harm": 0, "failed_lcb_safety": 0, "low_P_gain": 0, "negative_expected_utility": 0}

    def _make_prob_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        
    def _make_regressor_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

    def _train_prob_model(self, X, y, action_name, model_type):
        """Train a calibrated classifier or fallback to PriorProbabilityModel."""
        from sklearn.calibration import CalibratedClassifierCV

        unique_classes, counts = np.unique(y, return_counts=True)
        n_train = len(y)
        pos_count = dict(zip(unique_classes, counts)).get(1, 0)
        neg_count = dict(zip(unique_classes, counts)).get(0, 0)

        # Record class balance for diagnostics
        self.class_balance.setdefault(action_name, {})[f"{model_type}_positive"] = int(pos_count)
        self.class_balance[action_name][f"{model_type}_negative"] = int(neg_count)
        self.class_balance[action_name][f"{model_type}_n_train"] = int(n_train)

        if len(unique_classes) < 2:
            p_prior = pos_count / n_train if n_train > 0 else 0.5
            self.class_balance[action_name][f"{model_type}_model_used"] = "PriorProbabilityModel (single class)"
            return PriorProbabilityModel(p_prior)

        min_class_count = min(pos_count, neg_count)

        if min_class_count < 2:
            p_prior = pos_count / n_train
            self.class_balance[action_name][f"{model_type}_model_used"] = "PriorProbabilityModel (min count < 2)"
            return PriorProbabilityModel(p_prior)

        cv = min(3, min_class_count)

        try:
            calibrated = CalibratedClassifierCV(self._make_prob_model(), method='sigmoid', cv=cv)
            calibrated.fit(X, y)
            self.class_balance[action_name][f"{model_type}_model_used"] = f"CalibratedClassifierCV (cv={cv})"
            return calibrated
        except Exception as e:
            p_prior = pos_count / n_train
            self.class_balance[action_name][f"{model_type}_model_used"] = f"PriorProbabilityModel (fallback: {e})"
            return PriorProbabilityModel(p_prior)

    def train(self, training_data: PSafeTrainingData, val_split=0.2):
        from sklearn.preprocessing import StandardScaler

        X = training_data.feature_matrix
        n = len(X)
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        
        dense_ndcg = training_data.action_ndcg[Action.A0_DENSE.value]
        self.models = {}
        
        for a in Action:
            # FIX 5: Never train models for A0 Dense — it is the baseline
            if a == Action.A0_DENSE or a.value not in training_data.action_ndcg:
                continue
                
            a_ndcg = training_data.action_ndcg[a.value]
            a_lat = training_data.action_latency.get(a.value, np.full(n, ACTION_LATENCY[a]))
            
            delta_ndcg = a_ndcg - dense_ndcg
            y_gain = (delta_ndcg > self.epsilon_gain).astype(int)
            y_harm = (delta_ndcg < -self.epsilon_harm).astype(int)
            y_util = delta_ndcg - self.lambda_lat * a_lat - self.lambda_harm * y_harm

            action_name = ACTION_NAMES.get(a, str(a))

            # FIX 3: Use _train_prob_model which never returns None
            p_gain_model = self._train_prob_model(X_s, y_gain, action_name, "gain")
            p_harm_model = self._train_prob_model(X_s, y_harm, action_name, "harm")
            
            delta_regressor = self._make_regressor_model().fit(X_s, delta_ndcg)
            lat_regressor = self._make_regressor_model().fit(X_s, a_lat)
            
            ensemble = []
            if self.use_lcb_safety:
                for b in range(self.n_bootstrap):
                    idx = np.random.choice(n, n, replace=True)
                    m = self._make_regressor_model().fit(X_s[idx], y_util[idx])
                    ensemble.append(m)
                
            self.models[a.value] = {
                'p_gain': p_gain_model,   # FIX 3: never None
                'p_harm': p_harm_model,   # FIX 3: never None
                'delta': delta_regressor,
                'lat': lat_regressor,
                'ensemble': ensemble,
            }

        self.is_trained = True

    def save_class_balance(self, out_dir: str):
        """Save router_class_balance.json with diagnostic info."""
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "router_class_balance.json"), "w") as f:
            json.dump(self.class_balance, f, indent=4, default=str)

    def route(self, features: RoutingFeatures, is_test_batch: bool=False, current_hybrid_rate: float=0.0) -> PSafeDecision:
        if not self.is_trained:
            return PSafeDecision(query_id=features.query_id, action=Action.A0_DENSE)
            
        x = features.to_array(self.feature_names).reshape(1, -1)
        x_s = self.scaler.transform(x)
        
        # FIX 5: Dense always has utility=0, p_gain=0, p_harm=0
        best_action = Action.A0_DENSE
        best_util = 0.0
        best_decision = PSafeDecision(query_id=features.query_id, action=Action.A0_DENSE, final_decision_reason="fallback_to_dense")
        
        action_utils = {Action.A0_DENSE.value: 0.0}
        action_p_gain = {Action.A0_DENSE.value: 0.0}
        action_p_harm = {Action.A0_DENSE.value: 0.0}
        action_pred_delta = {Action.A0_DENSE.value: 0.0}
        action_pred_lat = {Action.A0_DENSE.value: ACTION_LATENCY[Action.A0_DENSE]}
        rejected = {}
        
        # for minimum hybrid rate fallback
        fallback_action = Action.A0_DENSE
        max_fallback_util = -9999.0
        fallback_decision = None
        
        for a in Action:
            if a == Action.A0_DENSE or a.value not in self.models:
                continue
                
            m = self.models[a.value]
            
            # FIX 3: p_gain/p_harm models are never None now
            p_gain = float(m['p_gain'].predict_proba(x_s)[0, 1])
            p_harm = float(m['p_harm'].predict_proba(x_s)[0, 1])
            pred_delta = float(m['delta'].predict(x_s)[0])
            pred_lat = float(m['lat'].predict(x_s)[0])
            
            expected_util = pred_delta - self.lambda_lat * pred_lat - self.lambda_harm * p_harm
            action_utils[a.value] = expected_util
            action_p_gain[a.value] = p_gain
            action_p_harm[a.value] = p_harm
            action_pred_delta[a.value] = pred_delta
            action_pred_lat[a.value] = pred_lat
            
            lcb_util = 0.0
            if self.use_lcb_safety and m['ensemble']:
                util_preds = np.array([em.predict(x_s)[0] for em in m['ensemble']])
                lcb_util = np.percentile(util_preds, 5)
            
            # fallback tracking
            if expected_util > max_fallback_util:
                max_fallback_util = expected_util
                fallback_action = a
                fallback_decision = PSafeDecision(
                    query_id=features.query_id, action=a, expected_utility=expected_util,
                    p_gain=p_gain, p_harm=p_harm, lcb_utility=lcb_util, final_decision_reason="highest_utility_fallback"
                )
                
            # Constraints
            if p_harm > self.harm_threshold:
                rejected[a.value] = "high_P_harm"
                self.rejected_counts["high_P_harm"] += 1
                continue
            if self.use_lcb_safety and lcb_util <= 0:
                rejected[a.value] = "failed_lcb_safety"
                self.rejected_counts["failed_lcb_safety"] += 1
                continue
            if p_gain < self.gain_threshold:
                rejected[a.value] = "low_P_gain"
                self.rejected_counts["low_P_gain"] += 1
                continue
            if expected_util <= 0:
                rejected[a.value] = "negative_expected_utility"
                self.rejected_counts["negative_expected_utility"] += 1
                continue
            
            if expected_util > best_util:
                best_util = expected_util
                best_action = a
                best_decision = PSafeDecision(
                    query_id=features.query_id, action=a, expected_utility=expected_util,
                    p_gain=p_gain, p_harm=p_harm, lcb_utility=lcb_util, final_decision_reason="highest_valid_utility"
                )
                
        # FIX 4: Force minimum hybrid rate ONLY if explicitly allowed and safe
        if (best_action == Action.A0_DENSE
                and current_hybrid_rate < self.min_hybrid_rate
                and fallback_action != Action.A0_DENSE
                and self.allow_forced_hybrid  # FIX 4: must be explicitly True
                and fallback_decision is not None
                and fallback_decision.expected_utility > 0
                and fallback_decision.p_harm <= self.harm_threshold + 0.10):
            best_decision = fallback_decision
            best_decision.final_decision_reason = "forced_hybrid_rate"
            
        best_decision.action_utilities = action_utils
        best_decision.action_p_gain = action_p_gain
        best_decision.action_p_harm = action_p_harm
        best_decision.action_pred_delta = action_pred_delta
        best_decision.action_pred_lat = action_pred_lat
        best_decision.features_used = features.features
        best_decision.rejected_reasons = rejected
        self.decisions.append(best_decision)
        return best_decision

    def get_stats(self):
        if not self.decisions: return {"router_type": "psafe"}
        actions = [d.action for d in self.decisions]
        n = len(actions)
        dist = {ACTION_NAMES[a]: sum(1 for x in actions if x == a) for a in Action}
        return {
            "router_type": "psafe", "total_queries": n,
            "dense_only_rate": sum(1 for a in actions if a == Action.A0_DENSE) / n,
            "hybrid_rate": sum(1 for a in actions if a != Action.A0_DENSE) / n,
            "action_distribution": dist,
            "rejected_counts": self.rejected_counts
        }

    def reset(self):
        self.decisions.clear()
        self.rejected_counts = {"high_P_harm": 0, "failed_lcb_safety": 0, "low_P_gain": 0, "negative_expected_utility": 0}

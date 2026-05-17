"""
B-P-SAFE-AMSR — Budgeted Probabilistic Safety-Aware Router
Canonical router for all final experiments.
"""
import numpy as np
import json
import os
import csv
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from dataclasses import dataclass, field

from ..retrievers.actions import Action, ACTION_NAMES


# ── Mode-specific default configs ────────────────────────────────────
MODE_DEFAULTS = {
    "lite": {
        "lambda_latency": 0.00020,
        "lambda_harm": 0.50,
        "lambda_recovery": 0.10,
        "lambda_candidate": 0.0001,
        "gain_threshold": 0.60,
        "harm_threshold": 0.20,
        "use_lcb_safety": True,
    },
    "balanced": {
        "lambda_latency": 0.00005,
        "lambda_harm": 0.25,
        "lambda_recovery": 0.25,
        "lambda_candidate": 0.0001,
        "gain_threshold": 0.40,
        "harm_threshold": 0.40,
        "use_lcb_safety": False,
    },
    "high_recall": {
        "lambda_latency": 0.00001,
        "lambda_harm": 0.15,
        "lambda_recovery": 0.35,
        "lambda_candidate": 0.00001,
        "gain_threshold": 0.25,
        "harm_threshold": 0.55,
        "use_lcb_safety": False,
    },
}


@dataclass
class PSafeDecision:
    query_id: str
    action: int
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
        """Deterministic prediction based on prior threshold."""
        return np.full(len(X), int(self.p >= 0.5))


class BPSafeRouter:
    """
    Budgeted Probabilistic Safety-Aware Adaptive Multi-Source Retrieval.
    """
    def __init__(self, mode="balanced", config=None):
        self.mode = mode
        self.config = config or {}

        # Resolve mode config: config file overrides > mode defaults
        defaults = MODE_DEFAULTS.get(self.mode, MODE_DEFAULTS["balanced"])
        mode_config = self.config.get("router_modes", {}).get(self.mode, {})

        self.gain_threshold = mode_config.get("gain_threshold", defaults["gain_threshold"])
        self.harm_threshold = mode_config.get("harm_threshold", defaults["harm_threshold"])
        self.lambda_latency = mode_config.get("lambda_latency", defaults["lambda_latency"])
        self.lambda_harm = mode_config.get("lambda_harm", defaults["lambda_harm"])
        self.lambda_candidate = mode_config.get("lambda_candidate", defaults["lambda_candidate"])
        self.lambda_recovery = mode_config.get("lambda_recovery", defaults["lambda_recovery"])
        self.use_lcb_safety = mode_config.get("use_lcb_safety", defaults["use_lcb_safety"])

        self.models_gain = {}
        self.models_harm = {}
        self.models_delta = {}
        self.models_lat = {}
        self.scaler = StandardScaler()

        self.actions = []
        self.diagnostics = {}
        self._action_predictions = []

    def _train_prob_model(self, X, y, action_name, model_type):
        """Train a calibrated classifier or fallback to PriorProbabilityModel."""
        unique_classes, counts = np.unique(y, return_counts=True)

        n_train = len(y)
        pos_count = dict(zip(unique_classes, counts)).get(1, 0)
        neg_count = dict(zip(unique_classes, counts)).get(0, 0)

        self.diagnostics.setdefault(action_name, {})[f"{model_type}_positive"] = int(pos_count)
        self.diagnostics[action_name][f"{model_type}_negative"] = int(neg_count)
        self.diagnostics[action_name][f"{model_type}_n_train"] = int(n_train)

        if len(unique_classes) < 2:
            p_prior = pos_count / n_train if n_train > 0 else 0.5
            self.diagnostics[action_name][f"{model_type}_used"] = "PriorProbabilityModel (single class)"
            return PriorProbabilityModel(p_prior)

        min_class_count = min(pos_count, neg_count)

        if min_class_count < 2:
            p_prior = pos_count / n_train
            self.diagnostics[action_name][f"{model_type}_used"] = "PriorProbabilityModel (min count < 2)"
            return PriorProbabilityModel(p_prior)

        cv = min(3, min_class_count)

        base_model = LogisticRegression(class_weight="balanced", max_iter=1000)
        try:
            calibrated = CalibratedClassifierCV(base_model, cv=cv, method="sigmoid")
            calibrated.fit(X, y)
            self.diagnostics[action_name][f"{model_type}_used"] = f"CalibratedClassifierCV (cv={cv})"
            return calibrated
        except Exception as e:
            p_prior = pos_count / n_train
            self.diagnostics[action_name][f"{model_type}_used"] = f"PriorProbabilityModel (fallback error: {e})"
            return PriorProbabilityModel(p_prior)

    def train(self, training_data):
        """Train the routing models. Skip A0 Dense (baseline has zero delta)."""
        X = self.scaler.fit_transform(training_data['features'])
        self.actions = training_data['actions']

        for a in self.actions:
            # Never train models for A0 Dense — it is the baseline
            if a == Action.A0_DENSE.value:
                continue

            y_delta = training_data['delta_ndcg'][a]
            y_lat = training_data['latency'][a]
            y_harm = training_data['harm'][a]
            y_gain = training_data['gain'][a]

            model_delta = Ridge()
            model_delta.fit(X, y_delta)
            self.models_delta[a] = model_delta

            model_lat = Ridge()
            model_lat.fit(X, y_lat)
            self.models_lat[a] = model_lat

            self.models_gain[a] = self._train_prob_model(X, y_gain, a, "gain")
            self.models_harm[a] = self._train_prob_model(X, y_harm, a, "harm")

    def tune_thresholds(self, val_data, out_dir=None):
        """FIX 6: Validation-only threshold tuning across ALL non-dense actions."""
        best_u = -float('inf')
        best_g = self.gain_threshold
        best_h = self.harm_threshold
        best_action_dist = {}

        X_val = self.scaler.transform(val_data['features'])
        y_delta_val = val_data.get('delta_ndcg', {})
        n_val = len(X_val)

        non_dense = [a for a in self.actions if a != Action.A0_DENSE.value and a in self.models_gain]
        if not non_dense or n_val == 0:
            return

        # Pre-compute predictions for all actions
        action_p_gains = {}
        action_p_harms = {}
        action_deltas_pred = {}
        action_deltas_actual = {}
        for a in non_dense:
            action_p_gains[a] = self.models_gain[a].predict_proba(X_val)[:, 1]
            action_p_harms[a] = self.models_harm[a].predict_proba(X_val)[:, 1]
            action_deltas_pred[a] = self.models_delta[a].predict(X_val)
            action_deltas_actual[a] = y_delta_val.get(a, np.zeros(n_val))

        gain_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
        harm_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

        for g_th in gain_thresholds:
            for h_th in harm_thresholds:
                realized_utility = np.zeros(n_val)
                selected_actions = np.full(n_val, Action.A0_DENSE.value)

                for qi in range(n_val):
                    best_qi_u = 0.0
                    for a in non_dense:
                        pg = action_p_gains[a][qi]
                        ph = action_p_harms[a][qi]
                        if pg < g_th or ph > h_th:
                            continue
                        pred_d = float(action_deltas_pred[a][qi])
                        u = pred_d - self.lambda_harm * ph + self.lambda_recovery * pg
                        if u > best_qi_u:
                            best_qi_u = u
                            selected_actions[qi] = a
                            realized_utility[qi] = float(action_deltas_actual[a][qi])

                mean_util = float(np.mean(realized_utility))
                if mean_util > best_u:
                    best_u = mean_util
                    best_g = g_th
                    best_h = h_th
                    from collections import Counter
                    best_action_dist = dict(Counter(selected_actions.tolist()))

        self.gain_threshold = best_g
        self.harm_threshold = best_h
        self.diagnostics["tuned_gain_threshold"] = best_g
        self.diagnostics["tuned_harm_threshold"] = best_h
        self.diagnostics["tuned_validation_utility"] = best_u
        self.diagnostics["tuned_action_distribution"] = best_action_dist

        # Save validation tuning results
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "validation_tuning.json"), "w") as f:
                json.dump({
                    "best_gain_threshold": best_g,
                    "best_harm_threshold": best_h,
                    "best_validation_utility": best_u,
                    "selected_action_distribution": best_action_dist,
                }, f, indent=4)

    def save_diagnostics(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "router_class_balance.json"), "w") as f:
            json.dump(self.diagnostics, f, indent=4, default=str)

        # Save mode config
        mode_cfg = {
            "mode": self.mode,
            "gain_threshold": self.gain_threshold,
            "harm_threshold": self.harm_threshold,
            "lambda_latency": self.lambda_latency,
            "lambda_harm": self.lambda_harm,
            "lambda_candidate": self.lambda_candidate,
            "lambda_recovery": self.lambda_recovery,
            "use_lcb_safety": self.use_lcb_safety,
        }
        with open(os.path.join(out_dir, "router_mode_config.json"), "w") as f:
            json.dump(mode_cfg, f, indent=4)

        with open(os.path.join(out_dir, "router_thresholds.json"), "w") as f:
            json.dump({
                "gain_threshold": self.gain_threshold,
                "harm_threshold": self.harm_threshold,
            }, f, indent=4)

        # Save action predictions if collected
        if self._action_predictions:
            with open(os.path.join(out_dir, "action_predictions.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._action_predictions[0].keys())
                writer.writeheader()
                writer.writerows(self._action_predictions)

    def route(self, features: np.ndarray, query_id: str,
              candidate_counts: Dict[int, int], split: str = "test") -> PSafeDecision:
        """FIX 4: candidate_count from actual counts, split recorded. FIX 5: LCB before negative utility."""
        X = self.scaler.transform(features.reshape(1, -1))

        action_utilities = {}
        action_p_gain = {}
        action_p_harm = {}
        action_pred_delta = {}
        action_pred_lat = {}
        rejected_reasons = {}

        # Dense fallback is ALWAYS A0_DENSE, never self.actions[0]
        best_a = Action.A0_DENSE.value
        best_u = 0.0

        action_pred_delta[best_a] = 0.0
        action_pred_lat[best_a] = 0.0
        action_p_gain[best_a] = 0.0
        action_p_harm[best_a] = 0.0
        action_utilities[best_a] = 0.0

        for a in self.actions:
            if a == Action.A0_DENSE.value:
                continue
            if a not in self.models_delta:
                rejected_reasons[a] = "missing_action_metrics"
                continue

            delta_pred = float(self.models_delta[a].predict(X)[0])
            lat_pred = float(self.models_lat[a].predict(X)[0])
            p_gain = float(self.models_gain[a].predict_proba(X)[0, 1])
            p_harm = float(self.models_harm[a].predict_proba(X)[0, 1])

            action_pred_delta[a] = delta_pred
            action_pred_lat[a] = lat_pred
            action_p_gain[a] = p_gain
            action_p_harm[a] = p_harm

            cand_count = candidate_counts.get(a, 0)

            u = (delta_pred
                 - self.lambda_latency * lat_pred
                 - self.lambda_harm * p_harm
                 - self.lambda_candidate * cand_count
                 + self.lambda_recovery * p_gain)

            # TODO: Replace simplified LCB with bootstrap/conformal uncertainty in future version.
            lcb_u = u  # FIX 5: Placeholder for proper LCB computation

            action_utilities[a] = u

            # FIX 5: Safety constraints — LCB checked before generic negative utility
            rejected = False
            reject_reason = ""
            if p_harm > self.harm_threshold:
                reject_reason = "high_P_harm"
                rejected = True
            elif p_gain < self.gain_threshold:
                reject_reason = "low_P_gain"
                rejected = True
            elif self.use_lcb_safety and lcb_u <= 0:
                reject_reason = "failed_lcb_safety"
                rejected = True
            elif u <= 0:
                reject_reason = "negative_expected_utility"
                rejected = True

            # Mode-specific soft override
            if rejected and self.mode == "balanced":
                if delta_pred > 0.02 and p_harm < self.harm_threshold + 0.10:
                    rejected = False
                    reject_reason = ""
            elif rejected and self.mode == "high_recall":
                if delta_pred > 0.005 and p_harm < self.harm_threshold + 0.20:
                    rejected = False
                    reject_reason = ""
            # lite: no override

            if rejected:
                rejected_reasons[a] = reject_reason
            else:
                if u > best_u:
                    best_u = u
                    best_a = a

        # FIX 4: Record prediction with actual candidate_count and split
        self._action_predictions.append({
            "query_id": query_id,
            "selected_action": best_a,
            "action_name": ACTION_NAMES.get(Action(best_a), str(best_a)),
            "expected_utility": round(best_u, 6),
            "p_gain": round(action_p_gain.get(best_a, 0.0), 4),
            "p_harm": round(action_p_harm.get(best_a, 0.0), 4),
            "pred_delta": round(action_pred_delta.get(best_a, 0.0), 6),
            "pred_latency": round(action_pred_lat.get(best_a, 0.0), 2),
            "candidate_count": candidate_counts.get(best_a, 0),
            "rejected_reason": rejected_reasons.get(best_a, ""),
            "mode": self.mode,
            "split": split,
        })

        return PSafeDecision(
            query_id=query_id,
            action=best_a,
            expected_utility=best_u,
            p_gain=action_p_gain.get(best_a, 0.0),
            p_harm=action_p_harm.get(best_a, 0.0),
            action_utilities=action_utilities,
            action_p_gain=action_p_gain,
            action_p_harm=action_p_harm,
            action_pred_delta=action_pred_delta,
            action_pred_lat=action_pred_lat,
            rejected_reasons=rejected_reasons,
            final_decision_reason=(
                "Highest utility passing safety constraints"
                if best_a != Action.A0_DENSE.value
                else "Fallback to Dense"
            ),
        )

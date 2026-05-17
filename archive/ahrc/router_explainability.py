"""
Safe-AMSR-SE v4 — Router Explainability
Feature importance, threshold sensitivity, and calibration analysis.
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExplainabilityReport:
    """Full explainability analysis for a trained router."""
    model_type: str
    feature_importances: List[Dict]          # [{name, importance, rank}]
    permutation_importances: List[Dict]       # [{name, importance_mean, importance_std}]
    threshold_curve: Dict                     # {thresholds, ndcg, safe_gain, hybrid_rate, losses}
    calibration_data: Dict                    # {bin_means, bin_true_fractions, n_per_bin}
    feature_correlations: Dict               # {feature_pairs, correlations}
    optimal_threshold: float
    optimal_metric: str


class RouterExplainer:
    """Analyze and explain router decisions."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def full_analysis(
        self,
        router,
        X_features: np.ndarray,
        dense_ndcg: np.ndarray,
        full_ndcg: np.ndarray,
        easy_mask: np.ndarray,
        output_dir: str = None,
    ) -> ExplainabilityReport:
        """Run complete explainability analysis."""
        fi = self._feature_importances(router)
        pi = self._permutation_importances(router, X_features, dense_ndcg, full_ndcg, easy_mask)
        tc = self._threshold_sensitivity(router, X_features, dense_ndcg, full_ndcg, easy_mask)
        cal = self._calibration_analysis(router, X_features, dense_ndcg, full_ndcg)
        corr = self._feature_correlations(X_features)

        # Find optimal threshold
        safe_gains = tc["safe_gain"]
        best_idx = int(np.argmax(safe_gains))
        opt_threshold = tc["thresholds"][best_idx]

        report = ExplainabilityReport(
            model_type=getattr(router, 'model_type', 'unknown'),
            feature_importances=fi,
            permutation_importances=pi,
            threshold_curve=tc,
            calibration_data=cal,
            feature_correlations=corr,
            optimal_threshold=float(opt_threshold),
            optimal_metric="safe_gain",
        )

        if output_dir:
            self._save_plots(report, output_dir)

        return report

    def _feature_importances(self, router) -> List[Dict]:
        """Extract native feature importances from the model."""
        if not hasattr(router, 'model') or router.model is None:
            return []

        importances = None
        if hasattr(router.model, 'feature_importances_'):
            importances = router.model.feature_importances_
        elif hasattr(router.model, 'coef_'):
            importances = np.abs(router.model.coef_[0])

        if importances is None:
            return []

        names = self.feature_names[:len(importances)]
        ranked = sorted(zip(names, importances), key=lambda x: -x[1])
        return [{"name": n, "importance": float(v), "rank": i + 1}
                for i, (n, v) in enumerate(ranked)]

    def _permutation_importances(
        self, router, X: np.ndarray,
        dense_ndcg: np.ndarray, full_ndcg: np.ndarray,
        easy_mask: np.ndarray, n_repeats: int = 10,
    ) -> List[Dict]:
        """Permutation-based feature importance (model-agnostic)."""
        if not hasattr(router, 'model') or router.model is None:
            return []
        if not hasattr(router, 'scaler') or router.scaler is None:
            return []

        X_scaled = router.scaler.transform(X)
        base_score = self._evaluate_router_score(router, X_scaled, dense_ndcg, full_ndcg, easy_mask)
        rng = np.random.default_rng(42)
        results = []

        for f_idx in range(X_scaled.shape[1]):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_scaled.copy()
                X_perm[:, f_idx] = rng.permutation(X_perm[:, f_idx])
                perm_score = self._evaluate_router_score_raw(
                    router.model, X_perm, router.safety_threshold,
                    dense_ndcg, full_ndcg, easy_mask
                )
                drops.append(base_score - perm_score)
            name = self.feature_names[f_idx] if f_idx < len(self.feature_names) else f"f_{f_idx}"
            results.append({
                "name": name,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            })

        results.sort(key=lambda x: -x["importance_mean"])
        return results

    def _evaluate_router_score(self, router, X_scaled, dense_ndcg, full_ndcg, easy_mask):
        """Compute SafeGain for the router on scaled features."""
        probs = router.model.predict_proba(X_scaled)[:, 1]
        use_hybrid = probs >= router.safety_threshold
        routed_ndcg = np.where(use_hybrid, full_ndcg, dense_ndcg)
        return self._safe_gain(routed_ndcg, dense_ndcg, easy_mask)

    def _evaluate_router_score_raw(self, model, X_scaled, threshold, dense_ndcg, full_ndcg, easy_mask):
        """Compute SafeGain given a model and pre-scaled features."""
        probs = model.predict_proba(X_scaled)[:, 1]
        use_hybrid = probs >= threshold
        routed_ndcg = np.where(use_hybrid, full_ndcg, dense_ndcg)
        return self._safe_gain(routed_ndcg, dense_ndcg, easy_mask)

    @staticmethod
    def _safe_gain(routed_ndcg, dense_ndcg, easy_mask):
        deltas = routed_ndcg - dense_ndcg
        hard_mask = ~easy_mask
        easy_deg = float(-np.mean(np.minimum(deltas[easy_mask], 0))) if np.sum(easy_mask) > 0 else 0
        hard_gain = float(np.mean(np.maximum(deltas[hard_mask], 0))) if np.sum(hard_mask) > 0 else 0
        return hard_gain - easy_deg

    def _threshold_sensitivity(
        self, router, X: np.ndarray,
        dense_ndcg: np.ndarray, full_ndcg: np.ndarray,
        easy_mask: np.ndarray,
    ) -> Dict:
        """Sweep threshold and compute metrics at each point."""
        if not hasattr(router, 'model') or router.model is None:
            return {"thresholds": [], "ndcg": [], "safe_gain": [], "hybrid_rate": [], "losses": []}

        X_scaled = router.scaler.transform(X)
        probs = router.model.predict_proba(X_scaled)[:, 1]
        thresholds = np.arange(0.05, 0.95, 0.02)
        ndcg_vals, sg_vals, hr_vals, loss_vals = [], [], [], []
        eps = 1e-8

        for t in thresholds:
            use_hybrid = probs >= t
            routed = np.where(use_hybrid, full_ndcg, dense_ndcg)
            deltas = routed - dense_ndcg
            ndcg_vals.append(float(np.mean(routed)))
            sg_vals.append(self._safe_gain(routed, dense_ndcg, easy_mask))
            hr_vals.append(float(np.mean(use_hybrid)))
            loss_vals.append(int(np.sum(deltas < -eps)))

        return {
            "thresholds": thresholds.tolist(),
            "ndcg": ndcg_vals,
            "safe_gain": sg_vals,
            "hybrid_rate": hr_vals,
            "losses": loss_vals,
        }

    def _calibration_analysis(self, router, X, dense_ndcg, full_ndcg, n_bins: int = 10) -> Dict:
        """Reliability diagram data."""
        if not hasattr(router, 'model') or router.model is None:
            return {}
        X_scaled = router.scaler.transform(X)
        probs = router.model.predict_proba(X_scaled)[:, 1]
        labels = (full_ndcg > dense_ndcg + 0.01).astype(int)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means, bin_true, bin_counts = [], [], []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            n = mask.sum()
            if n > 0:
                bin_means.append(float(probs[mask].mean()))
                bin_true.append(float(labels[mask].mean()))
                bin_counts.append(int(n))
        return {"bin_means": bin_means, "bin_true_fractions": bin_true, "n_per_bin": bin_counts}

    def _feature_correlations(self, X: np.ndarray, top_k: int = 10) -> Dict:
        """Correlation matrix for top features."""
        n_feats = min(X.shape[1], top_k, len(self.feature_names))
        corr_matrix = np.corrcoef(X[:, :n_feats].T)
        names = self.feature_names[:n_feats]
        pairs = []
        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                pairs.append({
                    "feature_a": names[i], "feature_b": names[j],
                    "correlation": float(corr_matrix[i, j]),
                })
        pairs.sort(key=lambda x: -abs(x["correlation"]))
        return {"top_correlations": pairs[:20], "feature_names": names,
                "matrix": corr_matrix.tolist()}

    def _save_plots(self, report: ExplainabilityReport, output_dir: str):
        """Generate explainability plots."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(output_dir, exist_ok=True)

        # 1. Feature importance bar chart
        if report.feature_importances:
            fig, ax = plt.subplots(figsize=(10, 6))
            names = [f["name"] for f in report.feature_importances[:15]]
            vals = [f["importance"] for f in report.feature_importances[:15]]
            ax.barh(names[::-1], vals[::-1], color='#2196F3', alpha=0.85)
            ax.set_xlabel("Importance"); ax.set_title("Router Feature Importances")
            ax.grid(axis='x', ls='--', alpha=0.5)
            fig.savefig(os.path.join(output_dir, "feature_importances.png"),
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # 2. Threshold sensitivity
        tc = report.threshold_curve
        if tc.get("thresholds"):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            ts = tc["thresholds"]
            ax1.plot(ts, tc["ndcg"], 'o-', color='#2196F3', ms=2, lw=1.5)
            ax1.axvline(report.optimal_threshold, color='red', ls='--', label=f"Opt={report.optimal_threshold:.2f}")
            ax1.set_xlabel("Threshold"); ax1.set_ylabel("nDCG@10"); ax1.set_title("nDCG vs Threshold")
            ax1.legend(); ax1.grid(True, ls='--', alpha=0.5)

            ax2.plot(ts, tc["safe_gain"], 'o-', color='#4CAF50', ms=2, lw=1.5)
            ax2.axvline(report.optimal_threshold, color='red', ls='--')
            ax2.set_xlabel("Threshold"); ax2.set_ylabel("SafeGain"); ax2.set_title("SafeGain vs Threshold")
            ax2.grid(True, ls='--', alpha=0.5)

            ax3.plot(ts, tc["hybrid_rate"], 'o-', color='#FF9800', ms=2, lw=1.5)
            ax3.axvline(report.optimal_threshold, color='red', ls='--')
            ax3.set_xlabel("Threshold"); ax3.set_ylabel("Hybrid Rate"); ax3.set_title("Activation Rate vs Threshold")
            ax3.grid(True, ls='--', alpha=0.5)

            fig.suptitle("Threshold Sensitivity Analysis", fontsize=14, y=1.02)
            fig.savefig(os.path.join(output_dir, "threshold_sensitivity.png"),
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # 3. Permutation importance
        if report.permutation_importances:
            fig, ax = plt.subplots(figsize=(10, 6))
            pi = report.permutation_importances[:15]
            names = [p["name"] for p in pi]
            means = [p["importance_mean"] for p in pi]
            stds = [p["importance_std"] for p in pi]
            ax.barh(names[::-1], means[::-1], xerr=stds[::-1], color='#9C27B0', alpha=0.85, capsize=3)
            ax.set_xlabel("SafeGain Drop"); ax.set_title("Permutation Feature Importances")
            ax.grid(axis='x', ls='--', alpha=0.5)
            fig.savefig(os.path.join(output_dir, "permutation_importances.png"),
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        print(f"   Explainability plots saved to {output_dir}")

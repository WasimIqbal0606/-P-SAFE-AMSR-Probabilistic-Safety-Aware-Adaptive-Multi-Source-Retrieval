"""
Safe-AMSR-SE v4 — Statistical Testing Suite
Full pairwise significance testing between all methods.

Tests per pair:
  1. Paired t-test
  2. Wilcoxon signed-rank
  3. Bootstrap 95% CI
  4. Permutation test
  5. Cohen's d effect size
  6. Win/Tie/Loss

Multiple testing:
  - Holm-Bonferroni correction across all pairwise comparisons
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats
import itertools


class StatisticalTester:
    """Publication-grade paired statistical tests for IR experiments."""

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000, n_permutation: int = 5000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutation = n_permutation

    def full_comparison(
        self,
        baseline_scores: np.ndarray,
        system_scores: np.ndarray,
        baseline_name: str = "Dense",
        system_name: str = "AMSR-SE",
        easy_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run all significance tests and produce a full comparison report."""
        baseline_scores = np.array(baseline_scores, dtype=np.float64)
        system_scores = np.array(system_scores, dtype=np.float64)
        deltas = system_scores - baseline_scores

        report = {
            "comparison": f"{system_name} vs {baseline_name}",
            "n_queries": len(deltas),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "std_delta": float(np.std(deltas)),
            "baseline_mean": float(np.mean(baseline_scores)),
            "system_mean": float(np.mean(system_scores)),
        }

        # Win/tie/loss
        eps = 1e-8
        report["wins"] = int(np.sum(deltas > eps))
        report["ties"] = int(np.sum(np.abs(deltas) <= eps))
        report["losses"] = int(np.sum(deltas < -eps))

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(baseline_scores) + np.var(system_scores)) / 2)
        if pooled_std > 0:
            cohens_d = float(np.mean(deltas) / pooled_std)
        else:
            cohens_d = 0.0
        effect_label = "negligible"
        if abs(cohens_d) >= 0.8: effect_label = "large"
        elif abs(cohens_d) >= 0.5: effect_label = "medium"
        elif abs(cohens_d) >= 0.2: effect_label = "small"
        report["effect_size"] = {
            "cohens_d": cohens_d,
            "magnitude": effect_label,
        }

        # 1. Paired t-test
        if np.std(deltas) > 0:
            t_stat, p_ttest = scipy_stats.ttest_rel(system_scores, baseline_scores)
        else:
            t_stat, p_ttest = 0.0, 1.0
        report["paired_ttest"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_ttest),
            "significant": bool(p_ttest < self.alpha),
        }

        # 2. Wilcoxon signed-rank test
        try:
            nonzero = deltas[np.abs(deltas) > eps]
            if len(nonzero) > 0:
                w_stat, p_wilcoxon = scipy_stats.wilcoxon(nonzero)
                report["wilcoxon"] = {
                    "w_statistic": float(w_stat),
                    "p_value": float(p_wilcoxon),
                    "significant": bool(p_wilcoxon < self.alpha),
                    "n_nonzero": len(nonzero),
                }
            else:
                report["wilcoxon"] = {"p_value": 1.0, "significant": False, "note": "all deltas zero"}
        except Exception as e:
            report["wilcoxon"] = {"p_value": 1.0, "error": str(e)}

        # 3. Bootstrap CI
        bootstrap_means = self._bootstrap_ci(deltas)
        ci_low = float(np.percentile(bootstrap_means, 2.5))
        ci_high = float(np.percentile(bootstrap_means, 97.5))
        report["bootstrap_ci"] = {
            "ci_low": ci_low, "ci_high": ci_high, "ci_level": 0.95,
            "significant": bool(ci_low > 0 or ci_high < 0),
            "n_bootstrap": self.n_bootstrap,
        }

        # 4. Permutation test
        p_perm = self._permutation_test(baseline_scores, system_scores)
        report["permutation_test"] = {
            "p_value": float(p_perm),
            "significant": bool(p_perm < self.alpha),
            "n_permutations": self.n_permutation,
        }

        # 5. Easy vs Hard breakdown
        if easy_mask is not None:
            easy_mask = np.array(easy_mask, dtype=bool)
            hard_mask = ~easy_mask
            report["easy_queries"] = self._group_stats(baseline_scores, system_scores, deltas, easy_mask)
            report["hard_queries"] = self._group_stats(baseline_scores, system_scores, deltas, hard_mask)

        return report

    def pairwise_comparison_matrix(
        self,
        method_ndcg: Dict[str, np.ndarray],
        easy_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run pairwise significance tests between ALL methods.
        Returns a structured matrix of results with Holm-Bonferroni correction.
        """
        methods = list(method_ndcg.keys())
        n_methods = len(methods)
        pairs = list(itertools.combinations(range(n_methods), 2))
        n_pairs = len(pairs)

        # Collect all pairwise p-values
        pairwise_results = {}
        raw_p_values = []
        pair_keys = []

        for i, j in pairs:
            m_a, m_b = methods[i], methods[j]
            key = f"{m_a} vs {m_b}"
            report = self.full_comparison(
                method_ndcg[m_a], method_ndcg[m_b],
                m_a, m_b, easy_mask
            )
            pairwise_results[key] = report
            raw_p_values.append(report["paired_ttest"]["p_value"])
            pair_keys.append(key)

        # Holm-Bonferroni correction
        corrected = self._holm_bonferroni(raw_p_values)
        for k, (key, p_corr) in enumerate(zip(pair_keys, corrected)):
            pairwise_results[key]["paired_ttest"]["p_corrected"] = float(p_corr)
            pairwise_results[key]["paired_ttest"]["significant_corrected"] = bool(p_corr < self.alpha)

        # Build significance matrix (n_methods x n_methods)
        sig_matrix = np.zeros((n_methods, n_methods))
        delta_matrix = np.zeros((n_methods, n_methods))
        for idx, (i, j) in enumerate(pairs):
            key = pair_keys[idx]
            p_corr = corrected[idx]
            delta = pairwise_results[key]["mean_delta"]
            sig_matrix[i, j] = 1 if p_corr < self.alpha and delta > 0 else (-1 if p_corr < self.alpha else 0)
            sig_matrix[j, i] = -sig_matrix[i, j]
            delta_matrix[i, j] = delta
            delta_matrix[j, i] = -delta

        return {
            "method_names": methods,
            "pairwise_results": pairwise_results,
            "significance_matrix": sig_matrix.tolist(),
            "delta_matrix": delta_matrix.tolist(),
            "n_comparisons": n_pairs,
            "correction": "holm_bonferroni",
        }

    def _holm_bonferroni(self, p_values: List[float]) -> List[float]:
        """Apply Holm-Bonferroni correction for multiple testing."""
        n = len(p_values)
        if n == 0:
            return []
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [0.0] * n
        for rank, (orig_idx, p) in enumerate(indexed):
            corrected[orig_idx] = min(1.0, p * (n - rank))
        # Enforce monotonicity
        for rank in range(1, n):
            orig_idx = indexed[rank][0]
            prev_idx = indexed[rank - 1][0]
            corrected[orig_idx] = max(corrected[orig_idx], corrected[prev_idx])
        return corrected

    def _group_stats(self, baseline, system, deltas, mask):
        n = int(np.sum(mask))
        if n == 0:
            return {"n": 0}
        eps = 1e-8
        return {
            "n": n,
            "baseline_mean": float(np.mean(baseline[mask])),
            "system_mean": float(np.mean(system[mask])),
            "mean_delta": float(np.mean(deltas[mask])),
            "degradation": bool(np.mean(deltas[mask]) < -eps),
            "improvement": bool(np.mean(deltas[mask]) > eps),
        }

    def _bootstrap_ci(self, deltas: np.ndarray) -> np.ndarray:
        """Paired bootstrap: resample deltas and compute means."""
        rng = np.random.default_rng(42)
        n = len(deltas)
        bootstrap_means = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = rng.choice(deltas, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)
        return bootstrap_means

    def _permutation_test(self, baseline: np.ndarray, system: np.ndarray) -> float:
        """Two-sided paired permutation test."""
        rng = np.random.default_rng(42)
        observed_diff = np.mean(system - baseline)
        n = len(baseline)
        count_extreme = 0
        for _ in range(self.n_permutation):
            signs = rng.choice([-1, 1], size=n)
            perm_diff = np.mean(signs * (system - baseline))
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1
        return count_extreme / self.n_permutation

    @staticmethod
    def format_report(report: Dict) -> str:
        """Pretty-print a statistical test report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  Statistical Significance: {report['comparison']}")
        lines.append("=" * 60)
        lines.append(f"  N queries:       {report['n_queries']}")
        lines.append(f"  Baseline mean:   {report['baseline_mean']:.4f}")
        lines.append(f"  System mean:     {report['system_mean']:.4f}")
        lines.append(f"  Mean Delta:      {report['mean_delta']:+.4f}")
        lines.append(f"  Median Delta:    {report['median_delta']:+.4f}")
        lines.append(f"  Win/Tie/Loss:    {report['wins']}/{report['ties']}/{report['losses']}")
        es = report.get("effect_size", {})
        lines.append(f"  Cohen's d:       {es.get('cohens_d', 0):.3f} ({es.get('magnitude', '?')})")
        tt = report.get("paired_ttest", {})
        sig = "YES" if tt.get("significant") else "NO"
        lines.append(f"  Paired t-test:   t={tt.get('t_statistic', 0):.3f}, p={tt.get('p_value', 1):.4e} -> {sig}")
        wt = report.get("wilcoxon", {})
        if "error" not in wt:
            sig = "YES" if wt.get("significant") else "NO"
            lines.append(f"  Wilcoxon:        p={wt.get('p_value', 1):.4e} -> {sig}")
        bc = report.get("bootstrap_ci", {})
        sig = "YES" if bc.get("significant") else "NO"
        lines.append(f"  Bootstrap 95%CI: [{bc.get('ci_low', 0):.4f}, {bc.get('ci_high', 0):.4f}] -> {sig}")
        pt = report.get("permutation_test", {})
        sig = "YES" if pt.get("significant") else "NO"
        lines.append(f"  Permutation:     p={pt.get('p_value', 1):.4e} -> {sig}")
        if "easy_queries" in report:
            eq = report["easy_queries"]
            hq = report["hard_queries"]
            lines.append(f"  Easy (n={eq['n']}): Delta={eq.get('mean_delta',0):+.4f}")
            lines.append(f"  Hard (n={hq['n']}): Delta={hq.get('mean_delta',0):+.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

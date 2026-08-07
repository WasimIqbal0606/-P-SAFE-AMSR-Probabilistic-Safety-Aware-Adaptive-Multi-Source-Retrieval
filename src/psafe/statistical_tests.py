"""
B-P-SAFE-AMSR — Canonical Statistical Testing Module
Publication-grade paired statistical testing, multiple-comparison control, and formal non-inferiority testing.

Tests per pair:
  1. Paired t-test
  2. Wilcoxon signed-rank test
  3. Paired Bootstrap 95% CI (5,000 draws)
  4. Sign-permutation test (2,000 draws)
  5. Cohen's d (pooled) and Cohen's dz (paired)
  6. Win / Tie / Loss counts
  7. Holm-Bonferroni multi-testing correction
  8. Formal Non-Inferiority Testing against Deep Hybrid (margin epsilon = 0.010)
  9. Multi-seed aggregation
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple
import itertools
import json
import os


def cohens_d_pooled(baseline, system):
    """Cohen's d with pooled standard deviation."""
    baseline = np.asarray(baseline, dtype=float)
    system = np.asarray(system, dtype=float)
    pooled_var = (np.var(baseline, ddof=1) + np.var(system, ddof=1)) / 2.0
    if pooled_var <= 0:
        return 0.0
    return float(np.mean(system - baseline) / np.sqrt(pooled_var))


def cohens_dz(deltas):
    """Paired Cohen's dz: mean(delta) / std(delta)."""
    deltas = np.asarray(deltas, dtype=float)
    d_std = np.std(deltas, ddof=1)
    if d_std <= 0:
        return 0.0
    return float(np.mean(deltas) / d_std)


def _effect_label(d_val):
    d_abs = abs(d_val)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down correction to a list of p-values.
    Guarantees monotonic adjustments and valid bounds in [0, 1].
    """
    p_values = np.asarray(p_values, dtype=np.float64)
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    corrected = np.zeros(m)
    for rank, idx in enumerate(sorted_indices):
        corrected[idx] = min(1.0, p_values[idx] * (m - rank))
    for rank in range(1, m):
        idx = sorted_indices[rank]
        prev_idx = sorted_indices[rank - 1]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])
    return [float(p) for p in corrected]


def evaluate_non_inferiority(
    system_scores: np.ndarray,
    reference_scores: np.ndarray,
    epsilon: float = 0.010,
    alpha: float = 0.05,
    n_bootstrap: int = 5000
) -> Dict:
    """
    Formal Non-Inferiority Analysis against reference (Deep Hybrid).
    H0: mean(system) - mean(reference) <= -epsilon  (System is inferior by more than epsilon)
    H1: mean(system) - mean(reference) >  -epsilon  (System is non-inferior within margin epsilon)

    Parameters:
      system_scores: query nDCG scores for candidate method (P-SAFE)
      reference_scores: query nDCG scores for reference method (Deep Hybrid)
      epsilon: non-inferiority margin (default 0.010 = 1.0% nDCG@10)
      alpha: significance level (default 0.05)
      n_bootstrap: bootstrap draws for empirical confidence bound
    """
    sys_arr = np.asarray(system_scores, dtype=np.float64)
    ref_arr = np.asarray(reference_scores, dtype=np.float64)
    deltas = sys_arr - ref_arr
    n = len(deltas)

    if n < 2:
        return {
            "mean_delta": float(np.mean(deltas)) if n > 0 else 0.0,
            "epsilon": float(epsilon),
            "non_inferiority_established": False,
            "p_value_non_inferiority": 1.0,
            "ci_lower_bound_95": -999.0,
            "conclusion": "insufficient samples"
        }

    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    se_delta = std_delta / np.sqrt(n)

    # 1. Parametric One-sided t-test against margin -epsilon
    if se_delta > 0:
        t_ni = (mean_delta - (-epsilon)) / se_delta
        df = n - 1
        p_val_ni = float(1.0 - scipy_stats.t.cdf(t_ni, df=df))
        t_crit = float(scipy_stats.t.ppf(1.0 - alpha, df=df))
        ci_lower_param = float(mean_delta - t_crit * se_delta)
    else:
        t_ni = 999.0 if mean_delta >= -epsilon else -999.0
        p_val_ni = 0.0 if mean_delta >= -epsilon else 1.0
        ci_lower_param = mean_delta

    # 2. Bootstrap One-sided Lower Bound (5th percentile of bootstrap mean deltas)
    rng = np.random.RandomState(42)
    boot_indices = rng.randint(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(deltas[boot_indices], axis=1)
    ci_lower_boot = float(np.percentile(boot_means, alpha * 100))

    non_inferior_established = bool(ci_lower_param > -epsilon and p_val_ni < alpha)

    if non_inferior_established:
        conclusion = f"Non-inferiority demonstrated: 95% lower bound ({ci_lower_param:+.4f}) remains above -{epsilon:.3f}"
    else:
        conclusion = f"Non-inferiority NOT demonstrated: 95% lower bound ({ci_lower_param:+.4f}) falls below -{epsilon:.3f}"

    return {
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "se_delta": se_delta,
        "n_queries": n,
        "epsilon_margin": float(epsilon),
        "t_statistic_ni": float(t_ni),
        "p_value_non_inferiority": float(p_val_ni),
        "ci_lower_bound_95_param": float(ci_lower_param),
        "ci_lower_bound_95_bootstrap": float(ci_lower_boot),
        "non_inferiority_established": non_inferior_established,
        "conclusion": conclusion,
    }


def get_significance_label(p_value, mean_delta, latency_saving=None):
    """Generate human-readable significance labels."""
    is_sig = p_value < 0.05
    if is_sig and mean_delta > 0:
        return "significant improvement"
    elif not is_sig and mean_delta > 0:
        return "positive but not significant"
    elif not is_sig and mean_delta <= 0 and latency_saving and latency_saving > 0.1:
        return "quality-preserving latency reduction"
    elif is_sig and mean_delta < 0 and latency_saving and latency_saving > 0.5:
        return "significant quality loss but high latency saving"
    elif not is_sig and mean_delta <= 0:
        return "protection mode"
    else:
        return "inconclusive"


class StatisticalTester:
    """Publication-grade paired statistical tests for IR experiments."""

    def __init__(self, alpha=0.05, n_bootstrap=5000, n_permutation=2000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutation = n_permutation

    def _bootstrap_ci(self, deltas: np.ndarray, seed: int = 42) -> np.ndarray:
        n = len(deltas)
        if n == 0:
            return np.array([0.0])
        rng = np.random.RandomState(seed)
        indices = rng.randint(0, n, size=(self.n_bootstrap, n))
        return np.mean(deltas[indices], axis=1)

    def _permutation_test(self, baseline: np.ndarray, system: np.ndarray, seed: int = 42) -> float:
        deltas = system - baseline
        n = len(deltas)
        if n == 0:
            return 1.0
        observed_mean = abs(float(np.mean(deltas)))
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1.0, 1.0], size=(self.n_permutation, n))
        perm_means = np.abs(np.mean(signs * deltas, axis=1))
        p_val = (np.sum(perm_means >= observed_mean) + 1) / (self.n_permutation + 1)
        return float(p_val)

    def _group_stats(self, baseline: np.ndarray, system: np.ndarray, deltas: np.ndarray, mask: np.ndarray) -> Dict:
        sub_base = baseline[mask]
        sub_sys = system[mask]
        sub_deltas = deltas[mask]
        if len(sub_deltas) == 0:
            return {"count": 0, "mean_delta": 0.0}
        return {
            "count": int(np.sum(mask)),
            "baseline_mean": float(np.mean(sub_base)),
            "system_mean": float(np.mean(sub_sys)),
            "mean_delta": float(np.mean(sub_deltas)),
            "std_delta": float(np.std(sub_deltas)),
        }

    def full_comparison(self, baseline_scores, system_scores,
                        baseline_name="Dense", system_name="B-P-SAFE",
                        easy_mask=None, latency_saving=None):
        """Run full significance testing suite and produce a structured report."""
        baseline = np.asarray(baseline_scores, dtype=np.float64)
        system = np.asarray(system_scores, dtype=np.float64)
        deltas = system - baseline
        eps = 1e-8

        report = {
            "comparison": f"{system_name} vs {baseline_name}",
            "n_queries": len(deltas),
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "std_delta": float(np.std(deltas)),
            "baseline_mean": float(np.mean(baseline)),
            "system_mean": float(np.mean(system)),
            "wins": int(np.sum(deltas > eps)),
            "ties": int(np.sum(np.abs(deltas) <= eps)),
            "losses": int(np.sum(deltas < -eps)),
        }

        # Effect size
        report["effect_size"] = {
            "cohens_d": cohens_d_pooled(baseline, system),
            "cohens_dz": cohens_dz(deltas),
            "magnitude": _effect_label(cohens_dz(deltas)),
        }

        # 1. Paired t-test
        if np.std(deltas) > 0:
            t_stat, p_ttest = scipy_stats.ttest_rel(system, baseline)
            if np.isnan(p_ttest):
                t_stat, p_ttest = 0.0, 1.0
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

        # 3. Bootstrap CI (5,000 draws)
        bootstrap_means = self._bootstrap_ci(deltas)
        ci_low = float(np.percentile(bootstrap_means, 2.5))
        ci_high = float(np.percentile(bootstrap_means, 97.5))
        report["bootstrap_ci"] = {
            "ci_low": ci_low, "ci_high": ci_high, "ci_level": 0.95,
            "significant": bool(ci_low > 0 or ci_high < 0),
            "n_bootstrap": self.n_bootstrap,
        }

        # 4. Permutation test (2,000 draws)
        p_perm = self._permutation_test(baseline, system)
        report["permutation_test"] = {
            "p_value": float(p_perm),
            "significant": bool(p_perm < self.alpha),
            "n_permutations": self.n_permutation,
        }

        # 5. Non-Inferiority test against baseline/reference (margin epsilon = 0.010)
        report["non_inferiority"] = evaluate_non_inferiority(
            system, baseline, epsilon=0.010, alpha=self.alpha, n_bootstrap=self.n_bootstrap
        )

        # 6. Easy vs Hard breakdown
        if easy_mask is not None:
            easy_mask = np.array(easy_mask, dtype=bool)
            report["easy_queries"] = self._group_stats(baseline, system, deltas, easy_mask)
            report["hard_queries"] = self._group_stats(baseline, system, deltas, ~easy_mask)

        return report

    def pairwise_comparison_matrix(self, method_ndcg: Dict[str, np.ndarray], easy_mask=None) -> Dict:
        """Run pairwise significance tests with Holm-Bonferroni correction."""
        methods = list(method_ndcg.keys())
        pairs = list(itertools.combinations(range(len(methods)), 2))

        pairwise_results = {}
        raw_p_values = []
        pair_keys = []

        for i, j in pairs:
            m_a, m_b = methods[i], methods[j]
            key = f"{m_a} vs {m_b}"
            report = self.full_comparison(method_ndcg[m_b], method_ndcg[m_a], m_b, m_a, easy_mask)
            pairwise_results[key] = report
            raw_p_values.append(report["paired_ttest"]["p_value"])
            pair_keys.append(key)

        corrected = holm_bonferroni_correction(raw_p_values)
        for k, (key, p_corr) in enumerate(zip(pair_keys, corrected)):
            pairwise_results[key]["paired_ttest"]["p_corrected"] = float(p_corr)
            pairwise_results[key]["paired_ttest"]["significant_corrected"] = bool(p_corr < self.alpha)

        return {
            "method_names": methods,
            "pairwise_results": pairwise_results,
            "n_comparisons": len(pairs),
            "correction": "holm_bonferroni",
        }

    def aggregate_multi_seed(self, seed_results_list: List[Dict], out_dir: str = ".") -> Dict:
        """
        Aggregate results across multiple seeds (e.g. 42, 123, 2026).
        """
        os.makedirs(out_dir, exist_ok=True)

        if not seed_results_list:
            res = {"error": "no seed results provided"}
            with open(os.path.join(out_dir, "multi_seed_summary.json"), "w") as f:
                json.dump(res, f, indent=4)
            return res

        keys = [k for k in seed_results_list[0].keys() if k != "seed"]
        summary = {}
        for k in keys:
            vals = [r[k] for r in seed_results_list if isinstance(r.get(k), (int, float))]
            if vals:
                summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "values": vals,
                }

        summary["n_seeds"] = len(seed_results_list)
        summary["seeds"] = [r.get("seed") for r in seed_results_list]

        with open(os.path.join(out_dir, "multi_seed_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        return summary

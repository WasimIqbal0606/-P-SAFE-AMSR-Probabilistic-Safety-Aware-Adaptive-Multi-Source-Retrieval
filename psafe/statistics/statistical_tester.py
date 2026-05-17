"""
B-P-SAFE-AMSR — Canonical Statistical Testing Module
Merged from archive/ahrc/statistical_tests.py (rich) + psafe stub.

Tests per pair:
  1. Paired t-test
  2. Wilcoxon signed-rank
  3. Bootstrap 95% CI
  4. Permutation test
  5. Cohen's d (pooled) and Cohen's dz (paired)
  6. Win/Tie/Loss
  7. Holm-Bonferroni correction
  8. Significance labels
  9. Multi-seed aggregation
"""
import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, List, Optional
import itertools
import json
import os


def cohens_d_pooled(baseline, system):
    """Cohen's d with pooled standard deviation."""
    pooled_std = np.sqrt((np.var(baseline) + np.var(system)) / 2)
    if pooled_std == 0:
        return 0.0
    return float(np.mean(system - baseline) / pooled_std)


def cohens_dz(deltas):
    """Paired Cohen's dz: mean(delta) / std(delta)."""
    d_std = np.std(deltas, ddof=1)
    if d_std == 0:
        return 0.0
    return float(np.mean(deltas) / d_std)


def holm_bonferroni_correction(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values."""
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
    return list(corrected)


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

    def __init__(self, alpha=0.05, n_bootstrap=10000, n_permutation=5000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutation = n_permutation

    def test_paired(self, baseline_scores, test_scores, latency_saving=None):
        """Quick paired test returning summary dict."""
        baseline = np.array(baseline_scores, dtype=np.float64)
        test = np.array(test_scores, dtype=np.float64)
        deltas = test - baseline
        mean_delta = float(np.mean(deltas))

        t_stat, p_val = scipy_stats.ttest_rel(test, baseline)
        if np.isnan(p_val):
            p_val = 1.0

        dz = cohens_dz(deltas)
        label = get_significance_label(p_val, mean_delta, latency_saving)

        return {
            "mean_delta": mean_delta,
            "p_value": float(p_val),
            "cohens_dz": dz,
            "label": label,
            "is_significant": bool(p_val < self.alpha),
        }

    def full_comparison(self, baseline_scores, system_scores,
                        baseline_name="Dense", system_name="B-P-SAFE",
                        easy_mask=None):
        """Run all significance tests and produce a full comparison report."""
        baseline = np.array(baseline_scores, dtype=np.float64)
        system = np.array(system_scores, dtype=np.float64)
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

        # Cohen's d (pooled) and dz (paired)
        report["effect_size"] = {
            "cohens_d": cohens_d_pooled(baseline, system),
            "cohens_dz": cohens_dz(deltas),
            "magnitude": _effect_label(cohens_d_pooled(baseline, system)),
        }

        # 1. Paired t-test
        if np.std(deltas) > 0:
            t_stat, p_ttest = scipy_stats.ttest_rel(system, baseline)
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
        p_perm = self._permutation_test(baseline, system)
        report["permutation_test"] = {
            "p_value": float(p_perm),
            "significant": bool(p_perm < self.alpha),
            "n_permutations": self.n_permutation,
        }

        # 5. Easy vs Hard breakdown
        if easy_mask is not None:
            easy_mask = np.array(easy_mask, dtype=bool)
            report["easy_queries"] = self._group_stats(baseline, system, deltas, easy_mask)
            report["hard_queries"] = self._group_stats(baseline, system, deltas, ~easy_mask)

        return report

    def pairwise_comparison_matrix(self, method_ndcg, easy_mask=None):
        """Run pairwise significance tests between ALL methods with Holm-Bonferroni."""
        methods = list(method_ndcg.keys())
        pairs = list(itertools.combinations(range(len(methods)), 2))

        pairwise_results = {}
        raw_p_values = []
        pair_keys = []

        for i, j in pairs:
            m_a, m_b = methods[i], methods[j]
            key = f"{m_a} vs {m_b}"
            report = self.full_comparison(method_ndcg[m_a], method_ndcg[m_b], m_a, m_b, easy_mask)
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

    def aggregate_multi_seed(self, seed_results_list, out_dir):
        """
        Aggregate results across multiple seeds (e.g. 42, 123, 2026).
        Each entry in seed_results_list is a dict with at least:
          - seed: int
          - dense_ndcg_mean: float
          - psafe_ndcg_mean: float
          - mean_delta: float
          - p_value: float
          - hybrid_activation_rate: float
        """
        os.makedirs(out_dir, exist_ok=True)

        if not seed_results_list:
            with open(os.path.join(out_dir, "multi_seed_summary.json"), "w") as f:
                json.dump({"error": "no seed results provided"}, f, indent=4)
            return {}

        keys = [k for k in seed_results_list[0].keys() if k != "seed"]
        summary = {}
        for k in keys:
            vals = [r[k] for r in seed_results_list if isinstance(r.get(k), (int, float))]
            if vals:
                summary[k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "values": vals,
                }

        summary["n_seeds"] = len(seed_results_list)
        summary["seeds"] = [r.get("seed") for r in seed_results_list]

        with open(os.path.join(out_dir, "multi_seed_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        return summary

    def save_results(self, results, out_dir, prefix=""):
        """Save test results to JSON files."""
        os.makedirs(out_dir, exist_ok=True)
        if isinstance(results, dict):
            fname = f"{prefix}statistical_tests.json" if prefix else "statistical_tests.json"
            with open(os.path.join(out_dir, fname), "w") as f:
                json.dump(results, f, indent=4, default=str)

    # ── Private helpers ──

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

    def _bootstrap_ci(self, deltas):
        rng = np.random.default_rng(42)
        n = len(deltas)
        means = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            means[i] = np.mean(rng.choice(deltas, size=n, replace=True))
        return means

    def _permutation_test(self, baseline, system):
        rng = np.random.default_rng(42)
        observed = np.mean(system - baseline)
        n = len(baseline)
        count = 0
        for _ in range(self.n_permutation):
            signs = rng.choice([-1, 1], size=n)
            perm_diff = np.mean(signs * (system - baseline))
            if abs(perm_diff) >= abs(observed):
                count += 1
        return count / self.n_permutation

    @staticmethod
    def format_report(report):
        """Pretty-print a statistical test report."""
        lines = ["=" * 60]
        lines.append(f"  Statistical Significance: {report['comparison']}")
        lines.append("=" * 60)
        lines.append(f"  N queries:       {report['n_queries']}")
        lines.append(f"  Mean Delta:      {report['mean_delta']:+.4f}")
        lines.append(f"  Win/Tie/Loss:    {report['wins']}/{report['ties']}/{report['losses']}")
        es = report.get("effect_size", {})
        lines.append(f"  Cohen's d:       {es.get('cohens_d', 0):.3f} ({es.get('magnitude', '?')})")
        lines.append(f"  Cohen's dz:      {es.get('cohens_dz', 0):.3f}")
        tt = report.get("paired_ttest", {})
        lines.append(f"  Paired t-test:   p={tt.get('p_value', 1):.4e} -> {'YES' if tt.get('significant') else 'NO'}")
        wt = report.get("wilcoxon", {})
        if "error" not in wt:
            lines.append(f"  Wilcoxon:        p={wt.get('p_value', 1):.4e} -> {'YES' if wt.get('significant') else 'NO'}")
        bc = report.get("bootstrap_ci", {})
        lines.append(f"  Bootstrap 95%CI: [{bc.get('ci_low', 0):.4f}, {bc.get('ci_high', 0):.4f}]")
        pt = report.get("permutation_test", {})
        lines.append(f"  Permutation:     p={pt.get('p_value', 1):.4e}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _effect_label(d):
    d = abs(d)
    if d >= 0.8: return "large"
    if d >= 0.5: return "medium"
    if d >= 0.2: return "small"
    return "negligible"

"""
Safe-AMSR-SE v3 — Table Generator
Produces CSV and LaTeX tables for publication.

Generates:
  - main_results.csv / main_results_latex.tex
  - ablation_table.csv
  - router_table.csv
"""

import os
import csv
import numpy as np
from typing import Dict, List, Any, Optional


class TableGenerator:
    """Generate publication-ready CSV and LaTeX tables."""

    @staticmethod
    def generate_main_results(
        results: Dict[str, Dict],
        safety_metrics: Dict[str, Dict],
        stat_reports: Dict[str, Dict],
        output_dir: str,
    ):
        """
        Generate the main results table.

        Columns: Method, nDCG@10, Recall@10, MRR, Latency_mean, Latency_p95,
                 Candidates_mean, Hybrid%, Easy_Δ, Hard_Δ, SafeGain, p_value_vs_dense
        """
        os.makedirs(output_dir, exist_ok=True)

        methods_order = [
            "Dense", "BM25", "Dense+BM25 RRF", "Dense+Graph",
            "Dense+BM25+Graph", "Full AMSR-SE",
            "Random Router", "Oracle",
            "Rule-based Safe", "Learned-LR Safe", "Learned-RF Safe",
            "Learned-GB Safe", "Cost-Aware Safe",
        ]

        # Filter to available methods
        available_methods = [m for m in methods_order if m in results]
        # Add any remaining methods not in order
        for m in results:
            if m not in available_methods and not m.startswith("_"):
                available_methods.append(m)

        headers = [
            "Method", "nDCG@10", "Recall@10", "MRR",
            "Lat_mean_ms", "Lat_p95_ms", "Cands_mean",
            "Hybrid%", "Easy_Δ", "Hard_Δ", "SafeGain",
            "p_vs_Dense",
        ]

        rows = []
        for method in available_methods:
            r = results.get(method, {})
            s = safety_metrics.get(method, {})
            st = stat_reports.get(method, {})

            ndcg10 = r.get("ndcg_at_k", {}).get("10", r.get("ndcg_at_k", {}).get(10, 0))
            recall10 = r.get("recall_at_k", {}).get("10", r.get("recall_at_k", {}).get(10, 0))
            mrr = r.get("mrr", 0)
            lat_mean = r.get("latency_mean_ms", 0)
            lat_p95 = r.get("latency_p95_ms", 0)
            cands = r.get("candidates_mean", 0)
            hybrid_pct = s.get("pct_routed_hybrid", 1.0) * 100
            easy_delta = s.get("easy_queries", {}).get("mean_delta", 0) if isinstance(s.get("easy_queries"), dict) else -s.get("easy_degradation_mean", 0)
            hard_delta = s.get("hard_queries", {}).get("mean_delta", 0) if isinstance(s.get("hard_queries"), dict) else s.get("hard_gain_mean", 0)
            safe_gain = s.get("safe_gain", 0)
            p_value = st.get("paired_ttest", {}).get("p_value", "—")

            rows.append([
                method,
                f"{ndcg10:.4f}",
                f"{recall10:.4f}",
                f"{mrr:.4f}",
                f"{lat_mean:.1f}",
                f"{lat_p95:.1f}",
                f"{cands:.0f}",
                f"{hybrid_pct:.1f}",
                f"{easy_delta:+.4f}" if isinstance(easy_delta, (int, float)) else "—",
                f"{hard_delta:+.4f}" if isinstance(hard_delta, (int, float)) else "—",
                f"{safe_gain:+.4f}" if isinstance(safe_gain, (int, float)) else "—",
                f"{p_value:.4e}" if isinstance(p_value, float) else str(p_value),
            ])

        # CSV
        csv_path = os.path.join(output_dir, "main_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"   📊 {csv_path}")

        # LaTeX
        latex_path = os.path.join(output_dir, "main_results_latex.tex")
        TableGenerator._write_latex(headers, rows, latex_path,
                                     caption="Safe-AMSR-SE v3: Main Results",
                                     label="tab:main_results")
        print(f"   📊 {latex_path}")

    @staticmethod
    def generate_ablation_table(
        ablation_results: Dict[str, Dict],
        output_dir: str,
    ):
        """Generate ablation study table."""
        os.makedirs(output_dir, exist_ok=True)

        headers = ["Configuration", "nDCG@10", "Recall@10", "MRR",
                    "Δ_nDCG@10", "Latency_ms"]

        # Get baseline
        baseline_ndcg = 0.0
        for key in ["Dense", "Dense Only"]:
            if key in ablation_results:
                r = ablation_results[key]
                baseline_ndcg = r.get("ndcg_at_k", {}).get("10", r.get("ndcg_at_k", {}).get(10, 0))
                break

        rows = []
        for method, r in ablation_results.items():
            if method.startswith("_"):
                continue
            ndcg = r.get("ndcg_at_k", {}).get("10", r.get("ndcg_at_k", {}).get(10, 0))
            recall = r.get("recall_at_k", {}).get("10", r.get("recall_at_k", {}).get(10, 0))
            mrr = r.get("mrr", 0)
            delta = ndcg - baseline_ndcg
            lat = r.get("latency_mean_ms", 0)
            rows.append([method, f"{ndcg:.4f}", f"{recall:.4f}", f"{mrr:.4f}",
                          f"{delta:+.4f}", f"{lat:.1f}"])

        csv_path = os.path.join(output_dir, "ablation_table.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"   📊 {csv_path}")

    @staticmethod
    def generate_router_table(
        router_stats: Dict[str, Dict],
        safety_metrics: Dict[str, Dict],
        output_dir: str,
    ):
        """Generate router comparison table."""
        os.makedirs(output_dir, exist_ok=True)

        headers = [
            "Router", "Type", "Hybrid%", "SafeGain",
            "Easy_Deg", "Hard_Gain", "Val_F1",
        ]

        rows = []
        for name, stats in router_stats.items():
            s = safety_metrics.get(name, {})
            train = stats.get("train_stats", {})
            rows.append([
                name,
                stats.get("router_type", "—"),
                f"{stats.get('hybrid_rate', s.get('pct_routed_hybrid', 0)) * 100:.1f}",
                f"{s.get('safe_gain', 0):+.4f}",
                f"{s.get('easy_degradation_mean', 0):.4f}",
                f"{s.get('hard_gain_mean', 0):+.4f}",
                f"{train.get('val_f1', 0):.3f}" if train.get('val_f1') else "—",
            ])

        csv_path = os.path.join(output_dir, "router_table.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows)
        print(f"   📊 {csv_path}")

    @staticmethod
    def _write_latex(
        headers: List[str],
        rows: List[List[str]],
        path: str,
        caption: str = "",
        label: str = "",
    ):
        """Write a LaTeX table."""
        n_cols = len(headers)
        col_spec = "l" + "r" * (n_cols - 1)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]

        # Header
        header_line = " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\"
        lines.append(header_line)
        lines.append(r"\midrule")

        # Data rows
        for row in rows:
            # Bold the best nDCG@10 value
            line = " & ".join(str(cell) for cell in row) + r" \\"
            lines.append(line)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

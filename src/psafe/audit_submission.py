"""Fail-closed semantic auditor for the paper evidence bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


EXPECTED_BASELINES = [
    "Dense-only",
    "Always-Hybrid",
    "Random",
    "Dense-margin",
    "Dense-entropy",
    "Regression-only",
    "Classification-only",
    "Oracle",
]
FORBIDDEN_BASELINES = {"BM25-disagreement", "Cost-only"}
REQUIRED_PRIMARY_FILES = [
    "extended_metrics.json",
    "statistical_tests.json",
    "per_query_metrics.csv",
    "latency_per_query.csv",
    "action_predictions.csv",
    "baseline_results.json",
    "baseline_per_query_metrics.csv",
    "reproducibility_manifest.json",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _hash_ids(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(item) for item in ids)).encode("utf-8")).hexdigest()


def _holm(raw: list[float]) -> list[float]:
    order = np.argsort(np.asarray(raw))
    adjusted = np.zeros(len(raw), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


class SubmissionAuditor:
    def __init__(self, config_path: str = "configs/paper_experiment.yaml", root_dir: str = "."):
        if Path(config_path).is_dir() and not config_path.endswith((".yaml", ".yml")):
            root_dir, config_path = config_path, "configs/paper_experiment.yaml"
        self.root = Path(root_dir).resolve()
        candidate = Path(config_path)
        self.config_path = candidate if candidate.is_absolute() else self.root / candidate
        self.config = _read_yaml(self.config_path) if self.config_path.exists() else {}
        self.datasets = list(self.config.get("datasets", ["scifact", "fiqa", "nfcorpus", "arguana"]))
        self.seeds = list(self.config.get("split_seeds", [42, 123, 2026]))
        self.modes = list(self.config.get("router_modes", {"lite": {}, "balanced": {}, "high_recall": {}}))
        self.failures: list[str] = []
        self.passes: list[str] = []
        self.warnings: list[str] = []

    def path(self, relative: str) -> Path:
        return self.root / relative

    def check(self, name: str, passed: bool, details: str) -> None:
        target = self.passes if passed else self.failures
        target.append(f"{'PASS' if passed else 'FAIL'}: {name} - {details}")

    def audit_config(self) -> None:
        required = {"datasets", "split_seeds", "router_modes", "statistics", "calibration", "baselines"}
        missing = sorted(required - set(self.config))
        reps = self.config.get("statistics", {}).get("matched_budget_random_repetitions")
        baselines = self.config.get("baselines", [])
        expected_config_baselines = EXPECTED_BASELINES[:3] + ["Matched-Budget-Random"] + EXPECTED_BASELINES[3:]
        ok = not missing and reps == 1000 and baselines == expected_config_baselines
        self.check("Canonical config", ok, f"missing={missing}, repetitions={reps}, baselines={baselines}")

    def _run_dirs(self) -> list[tuple[str, int, str, Path]]:
        return [
            (dataset, seed, mode, self.path(f"results/validated/{dataset}/seed_{seed}/{mode}"))
            for dataset in self.datasets
            for seed in self.seeds
            for mode in self.modes
        ]

    def audit_primary_evidence(self) -> None:
        issues: list[str] = []
        seen: set[tuple[str, int, str]] = set()
        for dataset, seed, mode, directory in self._run_dirs():
            missing = [name for name in REQUIRED_PRIMARY_FILES if not (directory / name).exists()]
            if missing:
                issues.append(f"{dataset}/seed_{seed}/{mode}: missing {missing}")
                continue
            manifest = _read_json(directory / "reproducibility_manifest.json")
            identity = (manifest.get("dataset"), manifest.get("seed"), manifest.get("mode"))
            expected = (dataset, seed, mode)
            if identity != expected:
                issues.append(f"{dataset}/seed_{seed}/{mode}: manifest identity {identity}")
            if identity in seen:
                issues.append(f"duplicate primary identity {identity}")
            seen.add(identity)
            metrics = _read_json(directory / "extended_metrics.json")
            rows = _read_csv(directory / "per_query_metrics.csv")
            ids = [row["query_id"] for row in rows]
            if len(ids) != len(set(ids)):
                issues.append(f"{dataset}/seed_{seed}/{mode}: duplicate test query IDs")
            for field, column in (
                ("dense_ndcg", "dense_ndcg"),
                ("best_hybrid_ndcg", "hybrid_ndcg"),
                ("psafe_ndcg", "psafe_ndcg"),
            ):
                observed = float(np.mean([float(row[column]) for row in rows]))
                if field not in metrics or not np.isclose(observed, metrics[field], atol=1e-10):
                    issues.append(f"{dataset}/seed_{seed}/{mode}: {field} mismatch")
        expected_count = len(self.datasets) * len(self.seeds) * len(self.modes)
        if len(seen) != expected_count:
            issues.append(f"unique manifests={len(seen)}, expected={expected_count}")
        self.check("Primary evidence", not issues, "; ".join(issues[:8]) or f"{expected_count} unique runs")

    def audit_split_provenance(self) -> None:
        issues: list[str] = []
        for dataset, seed, mode, directory in self._run_dirs():
            path = directory / "split_manifest.json"
            if not path.exists():
                issues.append(f"{dataset}/seed_{seed}/{mode}: split_manifest.json unavailable")
                continue
            data = _read_json(path)
            train = [str(x) for x in data.get("train_query_ids", [])]
            val = [str(x) for x in data.get("validation_query_ids", [])]
            test = [str(x) for x in data.get("test_query_ids", [])]
            if not train or not val or not test:
                issues.append(f"{dataset}/seed_{seed}/{mode}: explicit split IDs missing")
                continue
            if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
                issues.append(f"{dataset}/seed_{seed}/{mode}: split overlap")
            for name, ids in (("train", train), ("validation", val), ("test", test)):
                if data.get(f"{name}_query_ids_hash") != _hash_ids(ids):
                    issues.append(f"{dataset}/seed_{seed}/{mode}: {name} hash mismatch")
            observed_test = {row["query_id"] for row in _read_csv(directory / "per_query_metrics.csv")}
            if set(test) != observed_test:
                issues.append(f"{dataset}/seed_{seed}/{mode}: test IDs differ from per-query evidence")
        self.check("Split provenance", not issues, "; ".join(issues[:6]) or "explicit IDs, hashes, and disjointness verified")

    def audit_verified_baselines(self) -> None:
        compiled_path = self.path("results/baselines/comprehensive_baseline_results.json")
        issues: list[str] = []
        if not compiled_path.exists():
            self.check("Verified baselines", False, "compiled baseline artifact missing")
            return
        compiled = _read_json(compiled_path)
        for dataset in self.datasets:
            for seed in self.seeds:
                source_path = self.path(f"results/validated/{dataset}/seed_{seed}/balanced/baseline_results.json")
                source = _read_json(source_path)
                entry = compiled.get(dataset, {}).get(str(seed)) or compiled.get(dataset, {}).get(seed)
                if entry is None:
                    issues.append(f"{dataset}/{seed}: compiled entry missing")
                    continue
                if list(entry) != EXPECTED_BASELINES:
                    issues.append(f"{dataset}/{seed}: names={list(entry)}")
                if FORBIDDEN_BASELINES & set(entry):
                    issues.append(f"{dataset}/{seed}: unverified baseline retained")
                for name in EXPECTED_BASELINES:
                    if name not in source or entry.get(name) != source[name]:
                        issues.append(f"{dataset}/{seed}: {name} differs from source")
        self.check("Verified baselines", not issues, "; ".join(issues[:6]) or "eight executed baselines match per-run artifacts")

    @staticmethod
    def _matched_scores(dense: np.ndarray, hybrid: np.ndarray, target_k: int) -> np.ndarray:
        scores = np.empty(1000, dtype=float)
        n = len(dense)
        for repetition in range(1000):
            rng = np.random.RandomState(42 + repetition * 1000 + 7)
            selected = rng.permutation(n)[:target_k]
            routed = dense.copy()
            routed[selected] = hybrid[selected]
            scores[repetition] = float(np.mean(routed))
        return scores

    def audit_matched_random(self) -> None:
        path = self.path("results/baselines/matched_budget_random_results.json")
        if not path.exists():
            self.check("Matched-budget random", False, "artifact missing")
            return
        data = _read_json(path)
        issues: list[str] = []
        for dataset in self.datasets:
            for seed in self.seeds:
                for mode in ("balanced", "high_recall"):
                    entry = data.get(dataset, {}).get(str(seed), {}).get(mode)
                    if entry is None:
                        issues.append(f"{dataset}/{seed}/{mode}: missing")
                        continue
                    rows = _read_csv(self.path(f"results/validated/{dataset}/seed_{seed}/{mode}/per_query_metrics.csv"))
                    actions = _read_csv(self.path(f"results/validated/{dataset}/seed_{seed}/{mode}/action_predictions.csv"))
                    target_k = sum(str(row.get("selected_action")) in {"6", "A6_DEEP_HYBRID", "Deep Hybrid"} for row in actions)
                    dense = np.asarray([float(row["dense_ndcg"]) for row in rows])
                    hybrid = np.asarray([float(row["hybrid_ndcg"]) for row in rows])
                    psafe = float(np.mean([float(row["psafe_ndcg"]) for row in rows]))
                    if entry.get("n_repetitions") != 1000 or entry.get("n_queries") != len(rows):
                        issues.append(f"{dataset}/{seed}/{mode}: N/repetitions mismatch")
                    if entry.get("target_k") != target_k:
                        issues.append(f"{dataset}/{seed}/{mode}: target_k {entry.get('target_k')} != {target_k}")
                        continue
                    scores = self._matched_scores(dense, hybrid, target_k)
                    expected_p = float((1 + np.sum(scores >= psafe)) / 1001)
                    checks = [
                        np.isclose(entry.get("mean_ndcg"), np.mean(scores), atol=1e-12),
                        np.isclose(entry.get("std_ndcg"), np.std(scores, ddof=1), atol=1e-12),
                        np.isclose(entry.get("empirical_p_value"), expected_p, atol=1e-12),
                        np.isclose(entry.get("psafe_mean_ndcg"), psafe, atol=1e-12),
                    ]
                    if not all(checks):
                        issues.append(f"{dataset}/{seed}/{mode}: distribution or p-value not reproducible")
        self.check("Matched-budget random", not issues, "; ".join(issues[:6]) or "1000 allocations, exact K, and empirical p-values reproduced")

    def audit_calibration(self) -> None:
        path = self.path("results/calibration/calibration_metrics.json")
        issues: list[str] = []
        required = {
            "brier_score", "ece", "adaptive_ece", "auroc", "auprc",
            "calibration_slope", "calibration_intercept", "positive_rate", "n_samples", "n_positive",
        }
        if not path.exists():
            self.check("Calibration", False, "artifact missing")
            return
        data = _read_json(path)
        for dataset in self.datasets:
            for seed in self.seeds:
                for mode in ("balanced", "high_recall"):
                    for target in ("P_gain", "P_harm"):
                        entry = data.get(dataset, {}).get(str(seed), {}).get(mode, {}).get(target, {})
                        missing = required - set(entry)
                        if missing:
                            issues.append(f"{dataset}/{seed}/{mode}/{target}: missing {sorted(missing)}")
        manuscript = self.path("paper/manuscript.tex").read_text(encoding="utf-8")
        required_prose = "internal cross-validated sigmoid calibration on the training split"
        if required_prose not in manuscript or "calibration is fitted strictly on validation" in manuscript.lower():
            issues.append("manuscript calibration methodology contradicts implementation")
        self.check("Calibration", not issues, "; ".join(issues[:6]) or "metrics complete and prose matches training path")

    def audit_statistics(self) -> None:
        path = self.path("results/statistics/statistical_analysis.json")
        if not path.exists():
            self.check("Statistics", False, "artifact missing")
            return
        data = _read_json(path)
        issues: list[str] = []
        for family, raw_key, adjusted_key in (
            ("family_1_dense_improvement_holm", "raw_p_value", "holm_adjusted_p_value"),
            ("family_2_non_inferiority_holm", "raw_p_value_ni", "holm_adjusted_p_value_ni"),
        ):
            rows = data.get(family, [])
            expected = _holm([float(row[raw_key]) for row in rows])
            for row, value in zip(rows, expected):
                if not np.isclose(row[adjusted_key], value, atol=1e-12):
                    issues.append(f"{family}: Holm mismatch for {row.get('dataset')}/{row.get('mode')}")
                if family.startswith("family_2"):
                    decision = bool(row[adjusted_key] < 0.05 and row["ci_lower_bound_95"] > -row["epsilon"])
                    if row.get("non_inferiority_established") != decision:
                        issues.append(f"NI decision mismatch for {row.get('dataset')}/{row.get('mode')}")
        established = {
            (row["dataset"], row["mode"])
            for row in data.get("family_2_non_inferiority_holm", [])
            if row.get("non_inferiority_established")
        }
        if established != {("nfcorpus", "high_recall")}:
            issues.append(f"unexpected NI set {sorted(established)}")
        self.check("Statistics", not issues, "; ".join(issues[:6]) or "Holm and NI decisions reproduced")

    def audit_removed_evidence(self) -> None:
        active = [
            self.path("results/stability/fixed_split_training_seeds.json"),
            self.path("results/ablations/ablation_results.json"),
            self.path("paper/tables/stability_table.tex"),
            self.path("paper/tables/ablation_table.tex"),
            self.path("paper/figures/fig9_ablation_matrix.pdf"),
        ]
        present = [str(path.relative_to(self.root)) for path in active if path.exists()]
        self.check("Removed invalid evidence", not present, f"still active={present}" if present else "stability and feature ablations absent from active evidence")

    def audit_tables(self) -> None:
        generator = self.path("generate_paper_tables.py")
        text = generator.read_text(encoding="utf-8") if generator.exists() else ""
        issues = []
        if re.search(r"\.get\([^\n]+,\s*0(?:\.0)?\)", text):
            issues.append("table generator contains missing-to-zero fallback")
        if "def require(" not in text:
            issues.append("strict required-key helper absent")
        self.check("Table generation", not issues, "; ".join(issues) or "missing evidence raises")

    def audit_claim_registry_and_prose(self) -> None:
        registry_path = self.path("paper/claim_registry.json")
        issues: list[str] = []
        if not registry_path.exists():
            self.check("Claim registry and prose", False, "claim registry missing")
            return
        registry = _read_json(registry_path)
        allowed = {"VALIDATED", "UNSUPPORTED", "STALE", "REMOVED"}
        claims = registry.get("claims", [])
        for claim in claims:
            if claim.get("validation_status") not in allowed:
                issues.append(f"{claim.get('claim_id')}: invalid status")
            if not all(key in claim for key in (
                "claim_id", "manuscript_section", "metric", "dataset", "mode",
                "seed_scope", "source_artifact", "json_field_or_derivation", "validation_status",
            )):
                issues.append(f"{claim.get('claim_id')}: incomplete mapping")
        manuscript = self.path("paper/manuscript.tex").read_text(encoding="utf-8")
        readme = self.path("README.md").read_text(encoding="utf-8") if self.path("README.md").exists() else ""
        for label, content in (("manuscript", manuscript), ("README", readme)):
            for forbidden in ("ten independent training seeds", "compare 12 routing", "100 repetitions"):
                if forbidden.lower() in content.lower():
                    issues.append(f"{label}: stale phrase {forbidden!r}")
            if "arguana non-inferiority is established" in content.lower():
                issues.append(f"{label}: false ArguAna non-inferiority prose")
        arguana = _read_json(self.path("results/validated/arguana/seed_42/balanced/extended_metrics.json"))
        expected_arguana = f"{arguana['psafe_ndcg']:.4f}"
        for label, content in (("manuscript", manuscript), ("README", readme)):
            if expected_arguana not in content:
                issues.append(f"{label}: ArguAna P-SAFE value differs from source ({expected_arguana})")
        self.check("Claim registry and prose", not issues, "; ".join(issues[:8]) or "claim statuses and retained prose consistent")

    def run_full_audit(self) -> bool:
        self.audit_config()
        self.audit_primary_evidence()
        self.audit_split_provenance()
        self.audit_verified_baselines()
        self.audit_matched_random()
        self.audit_calibration()
        self.audit_statistics()
        self.audit_removed_evidence()
        self.audit_tables()
        self.audit_claim_registry_and_prose()
        for line in self.passes:
            print(line)
        for line in self.failures:
            print(line)
        print("SUBMISSION AUDIT: PASS" if not self.failures else "SUBMISSION AUDIT: FAIL")
        return not self.failures


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    success = SubmissionAuditor(root_dir=".").run_full_audit()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()

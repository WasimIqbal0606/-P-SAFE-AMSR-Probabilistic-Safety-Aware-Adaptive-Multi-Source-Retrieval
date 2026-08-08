"""Adversarial tests for the fail-closed submission auditor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from generate_paper_tables import require
from psafe.audit_submission import SubmissionAuditor, _hash_ids


REPO = Path(__file__).resolve().parents[1]


def copy_file(relative: str, root: Path) -> Path:
    source = REPO / relative
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def write_config(root: Path, datasets=None, seeds=None, modes=None) -> None:
    config = {
        "datasets": datasets or ["scifact"],
        "split_seeds": seeds or [42],
        "router_modes": {name: {} for name in (modes or ["lite", "balanced", "high_recall"])},
        "statistics": {"matched_budget_random_repetitions": 1000},
        "calibration": {"methods": ["sigmoid_platt"]},
        "baselines": [
            "Dense-only", "Always-Hybrid", "Random", "Matched-Budget-Random",
            "Dense-margin", "Dense-entropy", "Regression-only",
            "Classification-only", "Oracle",
        ],
    }
    path = root / "configs/paper_experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_repository_audit_fails_on_unavailable_historical_split_provenance():
    auditor = SubmissionAuditor(root_dir=str(REPO))
    assert auditor.run_full_audit() is False
    assert any("Split provenance" in failure for failure in auditor.failures)


def test_adversarial_train_test_overlap_fails(tmp_path: Path):
    write_config(tmp_path, modes=["lite"])
    run = tmp_path / "results/validated/scifact/seed_42/lite"
    run.mkdir(parents=True)
    copy_file("results/validated/scifact/seed_42/lite/per_query_metrics.csv", tmp_path)
    test_ids = ["100"]
    manifest = {
        "train_query_ids": test_ids,
        "validation_query_ids": ["validation-only"],
        "test_query_ids": test_ids,
        "train_query_ids_hash": _hash_ids(test_ids),
        "validation_query_ids_hash": _hash_ids(["validation-only"]),
        "test_query_ids_hash": _hash_ids(test_ids),
    }
    (run / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    auditor = SubmissionAuditor(root_dir=str(tmp_path))
    auditor.audit_split_provenance()
    assert any("split overlap" in failure for failure in auditor.failures)


def matched_root(tmp_path: Path) -> tuple[SubmissionAuditor, Path]:
    write_config(tmp_path, modes=["balanced", "high_recall"])
    for mode in ("balanced", "high_recall"):
        copy_file(f"results/validated/scifact/seed_42/{mode}/per_query_metrics.csv", tmp_path)
        copy_file(f"results/validated/scifact/seed_42/{mode}/action_predictions.csv", tmp_path)
    source = json.loads((REPO / "results/baselines/matched_budget_random_results.json").read_text(encoding="utf-8"))
    reduced = {"scifact": {"42": source["scifact"]["42"]}}
    path = tmp_path / "results/baselines/matched_budget_random_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reduced), encoding="utf-8")
    return SubmissionAuditor(root_dir=str(tmp_path)), path


@pytest.mark.parametrize("corruption", ["repetitions", "target_k", "p_value"])
def test_adversarial_matched_random_corruption_fails(tmp_path: Path, corruption: str):
    auditor, path = matched_root(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data["scifact"]["42"]["balanced"]
    if corruption == "repetitions":
        entry["n_repetitions"] = 999
    elif corruption == "target_k":
        entry["target_k"] += 1
    else:
        entry["empirical_p_value"] = 0.0
    path.write_text(json.dumps(data), encoding="utf-8")
    auditor.audit_matched_random()
    assert auditor.failures


def claim_root(tmp_path: Path) -> SubmissionAuditor:
    write_config(tmp_path)
    copy_file("paper/claim_registry.json", tmp_path)
    copy_file("paper/manuscript.tex", tmp_path)
    copy_file("README.md", tmp_path)
    copy_file("results/validated/arguana/seed_42/balanced/extended_metrics.json", tmp_path)
    return SubmissionAuditor(root_dir=str(tmp_path))


def test_adversarial_manuscript_number_mismatch_fails(tmp_path: Path):
    auditor = claim_root(tmp_path)
    manuscript = tmp_path / "paper/manuscript.tex"
    manuscript.write_text(manuscript.read_text(encoding="utf-8").replace("0.4069", "0.9999"), encoding="utf-8")
    auditor.audit_claim_registry_and_prose()
    assert any("ArguAna P-SAFE value differs" in failure for failure in auditor.failures)


def test_adversarial_arguana_noninferiority_prose_fails(tmp_path: Path):
    auditor = claim_root(tmp_path)
    manuscript = tmp_path / "paper/manuscript.tex"
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8") + "\nArguAna non-inferiority is established.\n",
        encoding="utf-8",
    )
    auditor.audit_claim_registry_and_prose()
    assert any("false ArguAna" in failure for failure in auditor.failures)


@pytest.mark.parametrize("artifact", ["ablations", "feature_train_equals_test", "stability"])
def test_adversarial_removed_scientific_artifact_fails(tmp_path: Path, artifact: str):
    if artifact == "stability":
        path = tmp_path / "results/stability/fixed_split_training_seeds.json"
        payload = {"scifact": {"per_seed_runs": [{"mean_ndcg": 0.5}] * 10}}
    else:
        path = tmp_path / "results/ablations/ablation_results.json"
        payload = {
            "scifact": {
                "Full B-P-SAFE": {"mean_ndcg": 0.0},
                "split_provenance": {"train_hash": "same", "validation_hash": "same", "test_hash": "same"},
            }
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    auditor = SubmissionAuditor(root_dir=str(tmp_path))
    auditor.audit_removed_evidence()
    assert auditor.failures


def test_adversarial_baseline_fallback_values_fail(tmp_path: Path):
    write_config(tmp_path, modes=["balanced"])
    source_path = copy_file("results/validated/scifact/seed_42/balanced/baseline_results.json", tmp_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    compiled = {"scifact": {"42": source}}
    compiled["scifact"]["42"]["Dense-margin"]["mean_latency"] = 400.0
    compiled["scifact"]["42"]["Dense-margin"]["hybrid_activation"] = 0.5
    path = tmp_path / "results/baselines/comprehensive_baseline_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compiled), encoding="utf-8")
    auditor = SubmissionAuditor(root_dir=str(tmp_path))
    auditor.audit_verified_baselines()
    assert auditor.failures


def test_table_missing_key_raises_instead_of_zero():
    with pytest.raises(KeyError):
        require({"dense_ndcg": 0.5}, "psafe_ndcg", "corrupted table fixture")


def test_adversarial_calibration_methodology_mismatch_fails(tmp_path: Path):
    write_config(tmp_path, modes=["balanced", "high_recall"])
    copy_file("results/calibration/calibration_metrics.json", tmp_path)
    manuscript = copy_file("paper/manuscript.tex", tmp_path)
    text = manuscript.read_text(encoding="utf-8").replace(
        "internal cross-validated sigmoid calibration on the training split",
        "calibration is fitted strictly on validation",
    )
    manuscript.write_text(text, encoding="utf-8")
    auditor = SubmissionAuditor(root_dir=str(tmp_path))
    auditor.audit_calibration()
    assert any("methodology contradicts" in failure for failure in auditor.failures)

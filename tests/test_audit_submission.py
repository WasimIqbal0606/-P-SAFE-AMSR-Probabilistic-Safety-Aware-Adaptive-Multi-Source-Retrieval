"""
Tests for canonical dataset configuration, submission audit, split isolation,
and hostile adversarial corruption fixtures.
"""
import pytest
import os
import shutil
import tempfile
import json
import yaml
import numpy as np
import pandas as pd

from psafe.audit_submission import SubmissionAuditor


def test_canonical_config_exists():
    assert os.path.exists("configs/paper_experiment.yaml")
    with open("configs/paper_experiment.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["datasets"] == ["scifact", "fiqa", "nfcorpus", "arguana"]
    assert cfg["split_seeds"] == [42, 123, 2026]
    assert "exploratory_datasets" in cfg
    assert "statistics" in cfg
    assert cfg["statistics"]["non_inferiority_margin"] == 0.010


def test_submission_auditor_pass():
    auditor = SubmissionAuditor()
    assert auditor.run_full_audit() is True


def test_no_split_overlap_in_validated_data():
    datasets = ["scifact", "fiqa", "nfcorpus", "arguana"]
    seeds = [42, 123, 2026]
    modes = ["lite", "balanced", "high_recall"]

    for ds in datasets:
        for seed in seeds:
            for mode in modes:
                ap_path = os.path.join("results/validated", ds, f"seed_{seed}", mode, "action_predictions.csv")
                if os.path.exists(ap_path):
                    df = pd.read_csv(ap_path)
                    if "split" in df.columns:
                        tr = set(df[df["split"] == "train"]["query_id"])
                        val = set(df[df["split"] == "val"]["query_id"])
                        te = set(df[df["split"] == "test"]["query_id"])

                        assert len(tr & val) == 0, f"train & val overlap in {ds}/seed_{seed}/{mode}"
                        assert len(tr & te) == 0, f"train & test overlap in {ds}/seed_{seed}/{mode}"
                        assert len(val & te) == 0, f"val & test overlap in {ds}/seed_{seed}/{mode}"


# ==============================================================================
# ADVERSARIAL AUDITOR TESTS: Intentionally Corrupted Fixtures Must Fail the Auditor
# ==============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary sandbox copying the verified directory structure."""
    tmp = tempfile.mkdtemp()
    
    # Copy essential folders
    for folder in ["configs", "results", "paper"]:
        if os.path.exists(folder):
            shutil.copytree(folder, os.path.join(tmp, folder))
            
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_adversarial_corrupted_ablation_control_fails(temp_workspace):
    """Fixture: Corrupted ablation full control (e.g. 0.64 instead of 0.6965) must FAIL."""
    abl_file = os.path.join(temp_workspace, "results/ablations/ablation_results.json")
    with open(abl_file) as f:
        data = json.load(f)
    # Corrupt SciFact Full B-P-SAFE control
    data["scifact"]["Full B-P-SAFE"]["mean_ndcg"] = 0.6465
    with open(abl_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Router Ablations" in f for f in auditor.failures)


def test_adversarial_corrupted_ablation_zero_har_fails(temp_workspace):
    """Fixture: Degenerate collapsed ablation control (0.0% HAR) must FAIL."""
    abl_file = os.path.join(temp_workspace, "results/ablations/ablation_results.json")
    with open(abl_file) as f:
        data = json.load(f)
    data["scifact"]["Full B-P-SAFE"]["hybrid_activation_rate"] = 0.0
    with open(abl_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False


def test_adversarial_matched_budget_k_mismatch_fails(temp_workspace):
    """Fixture: Matched-budget random using K+10 queries instead of exact target K must FAIL."""
    mb_file = os.path.join(temp_workspace, "results/baselines/matched_budget_random_results.json")
    with open(mb_file) as f:
        data = json.load(f)
    data["scifact"]["42"]["balanced"]["target_k"] = 999
    with open(mb_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Baseline Suite" in f for f in auditor.failures)


def test_adversarial_corrupted_calibration_brier_fails(temp_workspace):
    """Fixture: Out-of-bounds calibration metric (Brier score > 1.0) must FAIL."""
    cal_file = os.path.join(temp_workspace, "results/calibration/calibration_metrics.json")
    with open(cal_file) as f:
        data = json.load(f)
    data["scifact"]["42"]["balanced"]["P_gain"]["brier_score"] = 1.99
    with open(cal_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Calibration Diagnostics" in f for f in auditor.failures)


def test_adversarial_corrupted_holm_adjustment_fails(temp_workspace):
    """Fixture: Manually lowered Holm adjusted p-value (violating step-down bound) must FAIL."""
    stat_file = os.path.join(temp_workspace, "results/statistics/statistical_analysis.json")
    with open(stat_file) as f:
        data = json.load(f)
    # Violate raw <= adjusted bound
    data["family_1_dense_improvement_holm"][0]["holm_adjusted_p_value"] = 0.00001
    with open(stat_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Statistical & Non-Inferiority" in f for f in auditor.failures)


def test_adversarial_data_split_leakage_fails(temp_workspace):
    """Fixture: Synthetic overlap between train and test query IDs must FAIL."""
    ap_file = os.path.join(temp_workspace, "results/validated/scifact/seed_42/balanced/action_predictions.csv")
    df = pd.read_csv(ap_file)
    # Introduce deliberate leakage by appending a row marked split='train' with an ID that also exists in split='test'
    leaked_row = df.iloc[0].copy()
    leaked_row["split"] = "train"
    df = pd.concat([df, pd.DataFrame([leaked_row])], ignore_index=True)
    df.to_csv(ap_file, index=False)
    
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Split Leakage" in f for f in auditor.failures)


def test_adversarial_matched_random_repetitions_mismatch_fails(temp_workspace):
    """Fixture: Claiming 1000 repetitions when artifact has 100 must FAIL."""
    mb_file = os.path.join(temp_workspace, "results/baselines/matched_budget_random_results.json")
    with open(mb_file) as f:
        data = json.load(f)
    data["scifact"]["42"]["balanced"]["n_repetitions"] = 100
    with open(mb_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Baseline Suite" in f for f in auditor.failures)


def test_adversarial_non_inferiority_decision_mismatch_fails(temp_workspace):
    """Fixture: Fabricating Non-Inferiority on ArguAna when adjusted p > 0.05 must FAIL."""
    stat_file = os.path.join(temp_workspace, "results/statistics/statistical_analysis.json")
    with open(stat_file) as f:
        data = json.load(f)
    # Tamper decision to True for ArguAna where adjusted p is 0.1068
    for item in data["family_2_non_inferiority_holm"]:
        if item["dataset"] == "arguana":
            item["non_inferiority_established"] = True
    with open(stat_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Statistical & Non-Inferiority" in f for f in auditor.failures)


def test_adversarial_stability_missing_hashes_fails(temp_workspace):
    """Fixture: Stability runs without model/action hashes must FAIL."""
    stab_file = os.path.join(temp_workspace, "results/stability/fixed_split_training_seeds.json")
    with open(stab_file) as f:
        data = json.load(f)
    # Remove model hashes
    for r in data["scifact"]["per_seed_runs"]:
        r.pop("model_hash", None)
    with open(stab_file, "w") as f:
        json.dump(data, f, indent=4)
        
    auditor = SubmissionAuditor(root_dir=temp_workspace)
    assert auditor.run_full_audit() is False
    assert any("Training Stability" in f for f in auditor.failures)

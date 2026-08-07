"""
Tests for canonical dataset configuration, submission audit, and split isolation.
"""
import pytest
import os
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

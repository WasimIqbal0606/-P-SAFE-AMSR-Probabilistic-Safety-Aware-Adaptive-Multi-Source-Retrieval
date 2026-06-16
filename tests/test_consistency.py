def test_consistency_check():
    from psafe.experiment_core import check_consistency
    import tempfile
    import json
    import os
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "extended_metrics.json"), "w") as f:
            json.dump({"dense_ndcg": 0.5, "psafe_ndcg": 0.6}, f)
        with open(os.path.join(d, "statistical_tests.json"), "w") as f:
            json.dump({"P-SAFE vs Dense": {"baseline_mean": 0.5, "system_mean": 0.6}}, f)
        
        errors = check_consistency(d, 0.5, 0.6, 0.6, {}, 10)
        assert len(errors) == 0, "Should be consistent"

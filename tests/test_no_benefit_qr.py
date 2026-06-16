from psafe.metrics import calculate_extended_metrics


def test_quality_retention_does_not_explode_when_hybrid_has_no_gain():
    metrics = calculate_extended_metrics(
        dense_ndcg=0.42,
        hybrid_ndcg=0.42,
        psafe_ndcg=0.91,
        oracle_ndcg=0.91,
        always_on_hybrid_lat=1000,
        psafe_lat=50,
        always_on_hybrid_easy_deg=0,
        psafe_easy_deg=0,
        always_on_hybrid_hard_gain=0,
        psafe_hard_gain=0,
    )
    assert metrics["quality_retention_vs_best_hybrid"] == 0.0
    assert metrics["quality_retention_applicable"] is False
    assert metrics["dataset_regime"] == "no_benefit"

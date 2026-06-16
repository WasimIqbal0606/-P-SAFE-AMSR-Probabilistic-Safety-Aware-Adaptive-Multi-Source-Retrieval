def test_multi_seed_aggregate():
    from psafe.statistical_tests import StatisticalTester
    st = StatisticalTester()
    res = st.aggregate_multi_seed([
        {"seed": 42, "dense_ndcg_mean": 0.5, "psafe_ndcg_mean": 0.6},
        {"seed": 123, "dense_ndcg_mean": 0.5, "psafe_ndcg_mean": 0.62}
    ], out_dir=".")
    assert res["dense_ndcg_mean"]["mean"] == 0.5
    import os
    if os.path.exists("./multi_seed_summary.json"):
        os.remove("./multi_seed_summary.json")

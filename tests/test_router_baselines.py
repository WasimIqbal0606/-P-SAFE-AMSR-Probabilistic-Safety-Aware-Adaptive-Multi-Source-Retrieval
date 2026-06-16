def test_baselines_exist():
    from psafe.baselines import BASELINE_ROUTERS
    assert "Dense-only" in BASELINE_ROUTERS
    assert "Always-Hybrid" in BASELINE_ROUTERS
    assert "Random" in BASELINE_ROUTERS
    assert "Oracle" in BASELINE_ROUTERS

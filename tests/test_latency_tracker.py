def test_latency_tracker():
    from psafe.latency_tracker import LatencyTracker
    import time
    
    lt = LatencyTracker()
    lt.start_query("q1", "scifact")
    lt.record_metadata("seed", 42)
    lt.record_metadata("mode", "lite")
    with lt.track("dense_search"):
        time.sleep(0.01)
    lt.end_query()
    
    assert lt._records[0].seed == 42
    assert lt._records[0].mode == "lite"
